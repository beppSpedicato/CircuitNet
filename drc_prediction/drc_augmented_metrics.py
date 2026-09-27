import os
import os.path as osp
from functools import partial

import hydra
import numpy as np
import pandas as pd
import torch
from aim import Run
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric
from models.build_model import build_model

METRICS = ['NRMS', 'NRMS_design_with_violations', 'NRMS_nonzero', 'MAE', 'MAE_design_with_violations', 'MAE_nonzero']
# Reported in DRC violations per cell instead of normalized label units.
SCALED_METRICS = ['MAE', 'MAE_design_with_violations', 'MAE_nonzero']


@hydra.main(version_base=None, config_path="./config", config_name="drc_augmented_metrics_MSE_BN")
def evaluate(CFG):
    run = Run(experiment="drc_centralized_augmented_metrics")
    run['hparams'] = CFG
    if CFG.get('tag'):
        run.add_tag(CFG.tag)
    CFG = dict(CFG)

    CFG['ann_file'] = CFG['ann_file_test']
    CFG['test_mode'] = True
    label_scale = float(CFG.get('label_scale', 200))

    print('===> Loading datasets')
    dataset = build_dataset(CFG)

    print('===> Building model')
    model = build_model(dict(CFG))
    if not CFG.get('cpu', False):
        torch.cuda.set_device(CFG.get('gpu', 0))
        model = model.cuda()

    # Build metrics
    metrics = {k: build_metric(k) for k in METRICS}
    for k in SCALED_METRICS:
        metrics[k] = partial(metrics[k], label_scale=label_scale)
    avg_metrics = {k: 0 for k in METRICS}
    n_defined = {k: 0 for k in METRICS}

    rows = []
    with torch.no_grad():
        with tqdm(total=len(dataset)) as bar:
            for feature, label, label_path in dataset:
                if CFG.get('cpu', False):
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

                prediction = model(input)

                row = {
                    'sample': osp.splitext(osp.basename(label_path[0]))[0],
                    'violating_cells': int((target > 0).sum()),
                }
                for metric, metric_func in metrics.items():
                    metric_v = float(metric_func(target.cpu(), prediction.squeeze(1).cpu()))
                    row[metric] = metric_v
                    # NaN = undefined on this sample: kept in the CSV, left out of the average
                    if not np.isnan(metric_v):
                        avg_metrics[metric] += metric_v
                        n_defined[metric] += 1
                        run.track(
                            value=metric_v,
                            name=f"Test {metric} (dist)",
                            context={'subset': 'test', 'aggregation': 'distribution'},
                        )
                rows.append(row)

                bar.update(1)

    df = pd.DataFrame(rows)
    os.makedirs(CFG['save_path'], exist_ok=True)
    csv_path = osp.join(CFG['save_path'], 'augmented_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"===> Per-sample metrics saved to {csv_path}")

    n_with_violations = int((df['violating_cells'] > 0).sum())
    print(f"===> Samples with at least one violation: {n_with_violations}/{len(df)}")
    run.track(len(df), name='Test samples', context={'subset': 'test'})
    run.track(n_with_violations, name='Test samples with violations', context={'subset': 'test'})

    for metric, avg_metric in avg_metrics.items():
        n = n_defined[metric]
        avg = avg_metric / n if n else float('nan')
        print("===> Avg. {}: {:.4f} (over {} samples)".format(metric, avg, n))
        run.track(avg, name=f"Test Avg {metric}", context={'subset': 'test'})


if __name__ == "__main__":
    evaluate()
