# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import hydra
import numpy as np

from tqdm import tqdm
from aim import Run
from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
import torch

@hydra.main(version_base=None, config_path="./config", config_name="drc_test")
def test(CFG):
    run = Run(experiment="drc_centralized_test")
    run['hparams'] = CFG
    CFG = dict(CFG)

    CFG['ann_file'] = CFG['ann_file_test'] 
    CFG['test_mode'] = True

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(CFG)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(CFG)
    if not CFG.get('cpu', False):
        torch.cuda.set_device(CFG.get('gpu', 0))
        model = model.cuda()

    # Build metrics
    metrics = {k:build_metric(k) for k in CFG['eval_metric']}
    avg_metrics = {k:0 for k in CFG['eval_metric']}

    # Confusion matrix aggregation (one 2x2 CM per run, at CFG['threshold'])
    cm_threshold = CFG.get('threshold', 0.1)
    cm_total = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    count = 0
    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            if CFG.get('cpu', False):
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            prediction = model(input)
            for metric, metric_func in metrics.items():
                metric_v = metric_func(target.cpu(), prediction.squeeze(1).cpu())
                if metric_v != 1:
                    metric_v = float(metric_v)
                    avg_metrics[metric] += metric_v
                    # Track per-test metric value as a distribution
                    run.track(
                        value=metric_v,
                        name=f"Test {metric} (dist)",
                        context={'subset': 'test', 'aggregation': 'distribution'},
                    )

            if CFG['plot_roc']:
                save_path = osp.join(CFG['save_path'], 'test_result')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_name = osp.splitext(osp.basename(label_path[0]))[0]
                save_path = osp.join(save_path, f'{file_name}.npy')
                output_final = prediction.float().detach().cpu().numpy()
                np.save(save_path, output_final)
                count +=1

            bar.update(1)
    
    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))

    # eval roc&prc
    if CFG['plot_roc']:
        roc_metric, prc_numerator, tpr, fpr, precision, accuracy, tp, tn, fp, fn = build_roc_prc_metric(**CFG)
        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))
        print("===> TPR: {:.4f}".format(sum(tpr)/len(tpr)))
        print("===> FPR: {:.4f}".format((sum(fpr)/len(fpr))))
        print("===> Precision: {:.4f}".format(precision))
        print("===> Accuracy: {:.4f}".format(accuracy))
        print("===> PRC numerator: {:.4f}".format(prc_numerator))
        print("===> TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))

        run.track(accuracy, name='Test Accuracy', context={'subset': 'test'})

        run.track(tp, name='Confusion TP', context={'subset': 'test'})
        run.track(tn, name='Confusion TN', context={'subset': 'test'})
        run.track(fp, name='Confusion FP', context={'subset': 'test'})
        run.track(fn, name='Confusion FN', context={'subset': 'test'})

        for i in range(len(tpr)):
            run.track(tpr[i], name='ROC_TPR', step=i, context={'type': 'curve'})
        for i in range(len(fpr)):
            run.track(fpr[i], name='ROC_FPR', step=i, context={'type': 'curve'})

if __name__ == "__main__":
    test()
