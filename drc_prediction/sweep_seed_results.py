"""Aggregate the BS_GN seed sweep (run_sweep_seed.sh) from an Aim repo.

Mirrors show_experiment_result.ipynb, but every quantity is averaged (mean +- std)
over the seeds of the sweep. Figures and CSV tables are written to --out instead
of being shown, so it can run on a headless remote server.

    python sweep_seed_results.py --repo . --out sweep_seed_results
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aim import Repo
from aim.storage.context import Context

TRAIN_EXPERIMENT = 'drc_centralized_train_corrected'
TEST_EXPERIMENT = 'drc_centralized_test_corrected'
SWEEP_MARK = 'sweep_seed'          # substring of save_path set in sweep_seed_config/
SEED_RE = re.compile(r'_seed(\d+)')
FPR_GRID = np.linspace(0, 1, 501)


def sweep_runs(repo, experiment):
    """Return {seed: Run}, keeping the most recent run per seed."""
    query = f'run.active == False and run.archived == False and run.experiment == "{experiment}"'
    found = {}
    for wrapper in repo.query_runs(query=query, report_mode=0).iter_runs():
        run = wrapper.run
        hparams = run.get('hparams')
        save_path = str(hparams['save_path']) if hparams is not None else ''
        match = SEED_RE.search(save_path)
        if SWEEP_MARK not in save_path or match is None:
            continue
        seed = int(match.group(1))
        if seed not in found or run.creation_time > found[seed].creation_time:
            found[seed] = run
    return dict(sorted(found.items()))


def sorted_values(run, name, context):
    metric = run.get_metric(name, context=Context(context))
    if metric is None:
        return None
    ts = metric.timestamps.values_numpy()
    return metric.values.values_numpy()[np.argsort(ts)]


def scalar(run, name, context):
    vals = sorted_values(run, name, context)
    return None if vals is None or len(vals) == 0 else float(vals[0])


def roc_curve(run):
    tpr = sorted_values(run, 'ROC_TPR', {'type': 'curve'})
    fpr = sorted_values(run, 'ROC_FPR', {'type': 'curve'})
    if tpr is None or fpr is None:
        return None
    order = np.argsort(fpr, kind='stable')
    return fpr[order], tpr[order]


def auc(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def tpr_at_fpr(fpr, tpr, target=0.05):
    valid = np.where(fpr <= target)[0]
    if len(valid) == 0:
        return None, None
    return float(fpr[valid[-1]]), float(tpr[valid[-1]])


def mean_std_row(df, cols):
    return pd.DataFrame({'mean': df[cols].mean(), 'std': df[cols].std(ddof=0)})


def plot_mean_curve(ax, curves, title):
    """curves: list of 1D arrays (one per seed), truncated to the shortest."""
    n = min(len(c) for c in curves)
    arr = np.stack([c[:n] for c in curves])
    mean, std = arr.mean(0), arr.std(0)
    x = np.arange(n)
    for c in arr:
        ax.plot(x, c, color='0.7', linewidth=0.7)
    ax.plot(x, mean, color='C0', label='mean over seeds')
    ax.fill_between(x, mean - std, mean + std, color='C0', alpha=0.25, label='+- std')
    ax.set_title(f'{title}\nlast mean={mean[-1]:.4f} +- {std[-1]:.4f}')
    ax.legend()


def train_section(runs, out):
    pixel, epoch, rows = {}, {}, []
    for seed, run in runs.items():
        pixel[seed] = sorted_values(run, 'Pixel Loss', {'subset': 'global'})
        epoch[seed] = sorted_values(run, 'Epoch Loss', {'subset': 'global'})
        rows.append({'seed': seed, 'pixel_loss': pixel[seed][-1], 'epoch_loss': epoch[seed][-1]})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_mean_curve(axes[0], list(pixel.values()), 'Pixel Loss per iteration')
    plot_mean_curve(axes[1], list(epoch.values()), 'Epoch Loss (avg pixel loss per epoch)')
    fig.suptitle(f'BS_GN train, {len(runs)} seeds')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'train_loss.png'), dpi=150)
    plt.close(fig)

    summary = mean_std_row(df, ['pixel_loss', 'epoch_loss'])
    df.to_csv(os.path.join(out, 'train_per_seed.csv'), index=False)
    summary.to_csv(os.path.join(out, 'train_mean_std.csv'))
    print('\n== Train final losses, per seed ==\n', df.to_string(index=False))
    print('\n== Train final losses, mean/std ==\n', summary)


def test_section(runs, out):
    rows, curves, dists = [], {}, {'NRMS': [], 'SSIM': []}
    cms = []
    for seed, run in runs.items():
        row = {'seed': seed}
        for name in dists:
            vals = sorted_values(run, f'Test {name} (dist)', {'subset': 'test', 'aggregation': 'distribution'})
            vals = np.array([]) if vals is None else vals
            dists[name].append(vals)
            row[f'Avg {name}'] = float(np.mean(vals)) if len(vals) else None

        curve = roc_curve(run)
        if curve is not None:
            fpr, tpr = curve
            curves[seed] = (fpr, tpr)
            row['AUC ROC'] = auc(fpr, tpr)
            row['FPR <= 5%'], row['TPR @ FPR=5%'] = tpr_at_fpr(fpr, tpr)

        row['accuracy'] = scalar(run, 'Test Accuracy', {'subset': 'test'})
        row['precision'] = scalar(run, 'Test Precision', {'subset': 'test'})

        cm = [scalar(run, f'Confusion {k}', {'subset': 'test'}) for k in ('TN', 'FP', 'FN', 'TP')]
        if None not in cm:
            cm = np.array(cm, dtype=float).reshape(2, 2)
            cms.append(cm / cm.sum())
            tn, fp, fn, tp = cm.ravel()
            row['recall'] = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c != 'seed']
    summary = mean_std_row(df, metric_cols)
    df.to_csv(os.path.join(out, 'test_per_seed.csv'), index=False)
    summary.to_csv(os.path.join(out, 'test_mean_std.csv'))
    print('\n== Test metrics, per seed ==\n', df.to_string(index=False))
    print('\n== Test metrics, mean/std over seeds ==\n', summary)

    # NRMS / SSIM distributions pooled over all seeds
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, dists):
        pooled = np.concatenate(dists[name]) if any(len(v) for v in dists[name]) else np.array([])
        ax.hist(pooled, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'{name} distribution (pooled over seeds)')
        if len(pooled):
            ax.text(0.98, 0.98, f'mean={pooled.mean():.4f}\nstd={pooled.std():.4f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7'))
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'test_metric_distributions.png'), dpi=150)
    plt.close(fig)

    # Mean ROC: interpolate every seed on a common FPR grid
    if curves:
        fig, ax = plt.subplots(figsize=(8, 6))
        interp = []
        for seed, (fpr, tpr) in curves.items():
            uniq, idx = np.unique(fpr, return_index=True)
            t = np.interp(FPR_GRID, uniq, tpr[idx])
            interp.append(t)
            ax.plot(FPR_GRID, t, color='0.7', linewidth=0.8, label=f'seed {seed} (AUC={auc(fpr, tpr):.4f})')
        interp = np.stack(interp)
        mean, std = interp.mean(0), interp.std(0)
        ax.plot(FPR_GRID, mean, color='C0', label=f'mean (AUC={auc(FPR_GRID, mean):.4f})')
        ax.fill_between(FPR_GRID, mean - std, mean + std, color='C0', alpha=0.25)
        ax.plot([0, 1], [0, 1], linestyle='--', color='k')
        ax.set(title='ROC curves (mean +- std over seeds)', xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'test_roc_mean.png'), dpi=150)
        plt.close(fig)

    # Mean confusion matrix (fractions, averaged over seeds)
    if cms:
        mean_cm = np.mean(cms, axis=0)
        std_cm = np.std(cms, axis=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mean_cm, cmap='Blues')
        ax.set_xticks([0, 1], ['Pred 0', 'Pred 1'])
        ax.set_yticks([0, 1], ['True 0', 'True 1'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{100 * mean_cm[i, j]:.2f}%\n+- {100 * std_cm[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if mean_cm[i, j] > mean_cm.max() / 2 else 'black')
        ax.set_title(f'Mean confusion matrix ({len(cms)} seeds)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'test_confusion_mean.png'), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=os.getcwd(), help='directory containing the .aim repo')
    parser.add_argument('--out', default='sweep_seed_results', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    repo = Repo(path=args.repo)

    train_runs = sweep_runs(repo, TRAIN_EXPERIMENT)
    test_runs = sweep_runs(repo, TEST_EXPERIMENT)
    print(f'train seeds found: {list(train_runs)}  |  test seeds found: {list(test_runs)}')

    if train_runs:
        train_section(train_runs, args.out)
    else:
        print('No finished train runs found for the sweep.')
    if test_runs:
        test_section(test_runs, args.out)
    else:
        print('No finished test runs found for the sweep.')
    print(f'\nOutputs written to {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
