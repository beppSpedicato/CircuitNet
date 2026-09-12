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
import pandas as pd

@hydra.main(version_base=None, config_path="./config", config_name="drc_test_correct")
def test(CFG):
    run = Run(experiment="drc_centralized_test_corrected")
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
        roc_metric, prc_numerator = build_roc_prc_metric(**CFG)
        

        csv_file = osp.join(CFG['save_path'], 'roc_prc.csv')
        df = pd.read_csv(csv_file, header=None, names=["threshold", "id", "tn", "fp", "fn", "tp"])
        t = df
        no_negatives      = (t["fp"] == 0) & (t["tn"] == 0)
        no_positives      = (t["tp"] == 0) & (t["fn"] == 0)
        no_pred_positives = (t["tp"] == 0) & (t["fp"] == 0)
        df = t[~(no_negatives | no_positives | no_pred_positives)]
        df["pos"] = df["tp"] + df["fn"]
        df["neg"] = df["fp"] + df["tn"]
        valid = df[(df["pos"] > 0) & (df["neg"] > 0)].copy()
        valid["tpr"] = valid["tp"] / valid["pos"]
        valid["fpr"] = valid["fp"] / valid["neg"]
        macro = valid.groupby("threshold")[["fpr", "tpr"]].mean()
        df_filtered = valid[valid["threshold"] == 0.1].copy()
        accuracy = (df_filtered['tp'].sum() + df_filtered['tn'].sum()) / (df_filtered['tp'].sum() + df_filtered['tn'].sum() + df_filtered['fp'].sum() + df_filtered['fn'].sum())
        precision = df_filtered['tp'].sum() / (df_filtered['tp'].sum() + df_filtered['fp'].sum())

        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))
        print("===> Precision: {:.4f}".format(precision))
        print(f"===> Accuracy @ score>={CFG['threshold']}: {accuracy:.4f}")
        print("===> PRC numerator: {:.4f}".format(prc_numerator))

        run.track(accuracy, name='Test Accuracy', context={'subset': 'test'})
        run.track(precision, name='Test Precision', context={'subset': 'test'})
        run.track(df_filtered['tp'].sum(), name='Confusion TP', context={'subset': 'test'})
        run.track(df_filtered['tn'].sum(), name='Confusion TN', context={'subset': 'test'})
        run.track(df_filtered['fp'].sum(), name='Confusion FP', context={'subset': 'test'})
        run.track(df_filtered['fn'].sum(), name='Confusion FN', context={'subset': 'test'})

        for i in range(len(macro['tpr'])):
            run.track(macro['tpr'].iloc[i], name='ROC_TPR', step=i, context={'type': 'curve'})
        for i in range(len(macro['fpr'])):
            run.track(macro['fpr'].iloc[i], name='ROC_FPR', step=i, context={'type': 'curve'})


if __name__ == "__main__":
    test()
