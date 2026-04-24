# Copyright 2022 CircuitNet. All rights reserved.

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from utils.metrics import build_metric
from models.build_model import build_model
from math import cos, pi
import os
from aim import Run
import hydra
import omegaconf


def checkpoint(model, epoch, save_path, run: Run):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    run.log_info("Checkpoint saved to {}".format(model_out_path))
        
class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]


def validate(model, loss_fn, metrics, val_loader, device):
    model.eval()

    avg_metrics = {k: 0.0 for k in metrics.keys()}
    avg_loss = 0.0
    n = 0

    with torch.no_grad():
        for feature, label, _ in val_loader:
            input = feature.to(device)
            target = label.to(device)

            prediction = model(input)
            avg_loss += loss_fn(prediction, target).item()

            pred_cpu = prediction.squeeze(1).detach().cpu()
            tgt_cpu = target.cpu()
            for name, fn in metrics.items():
                v = fn(tgt_cpu, pred_cpu)
                if v != 1:
                    avg_metrics[name] += float(v)

            n += 1

    if n > 0:
        avg_loss /= n
        for k in avg_metrics:
            avg_metrics[k] /= n

    model.train()
    return avg_loss, avg_metrics


@hydra.main(version_base=None, config_path="./config", config_name="drc_train")
def train(CFG: omegaconf.dictconfig.DictConfig):
    run = Run(experiment="drc_centralized_train")
    run['hparams'] = CFG

    if not os.path.exists(CFG.save_path):
        os.makedirs(CFG.save_path)

    CFG = dict(CFG)

    print('===> Loading datasets')
    # Train loader (iterable infinite stream)
    train_opt = dict(CFG)
    train_opt['ann_file'] = CFG['ann_file_train']
    train_opt['test_mode'] = False
    train_loader = build_dataset(train_opt)

    # Val loader (finite, deterministic)
    val_opt = dict(CFG)
    val_opt['ann_file'] = CFG['ann_file_val']
    val_opt['test_mode'] = True
    val_loader = build_dataset(val_opt)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(CFG)

    # set device
    device = torch.device('cpu' if CFG.get('cpu', False) else 'cuda')
    if not CFG.get('cpu', False):
        torch.cuda.set_device(CFG.get('gpu', 0))
        model = model.cuda()
    
    # Build loss
    loss = build_loss(CFG)

    # Build validation metrics
    metrics = {k: build_metric(k) for k in CFG['eval_metric']}

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'],  betas=(0.9, 0.999), weight_decay=CFG['weight_decay'])

    # Build lr scheduler
    cosine_lr = CosineRestartLr(CFG['lr'], [CFG['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = CFG['save_freq']
    
    epoch = 0

    while iter_num < CFG['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in train_loader:
                input, target = feature.to(device), label.to(device)

                regular_lr = cosine_lr.get_regular_lr(iter_num)
                cosine_lr._set_lr(optimizer, regular_lr)

                prediction = model(input)

                optimizer.zero_grad()
                pixel_loss = loss(prediction, target)

                run.track(
                    value=pixel_loss.item(),
                    name="Pixel Loss",
                    context={'subset': 'global'},
                    step=iter_num,
                )

                epoch_loss += pixel_loss.item()
                
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1
                
                bar.update(1)

                if iter_num % print_freq == 0:
                    break

        
        print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, CFG['max_iters'], epoch_loss / print_freq))
        run.track(
            value=epoch_loss / print_freq,
            name="Epoch Loss",
            context={'subset': 'global'},
            step=epoch,
        )
        
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, CFG['save_path'], run)

        if epoch % CFG.get('eval_freq_epochs', 1) == 0:
            val_loss, val_metrics = validate(model, loss, metrics, val_loader, device)
            run.track(
                value=val_loss,
                name="Val Pixel Loss",
                context={'subset': 'val'},
                step=epoch,
            )
            for k, v in val_metrics.items():
                run.track(
                    value=v,
                    name=f"Val {k}",
                    context={'subset': 'val'},
                    step=epoch,
                )
            print(
                "===> Val[epoch {}]: loss {:.4f} ".format(
                    epoch, val_loss
                ) + " ".join([f"{k}:{val_metrics[k]:.4f}" for k in val_metrics])
            )

        epoch += 1
        epoch_loss = 0

if __name__ == "__main__":
    train()
