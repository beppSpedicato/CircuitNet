# Copyright 2022 CircuitNet. All rights reserved.

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
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

@hydra.main(version_base=None, config_path="./config", config_name="drc_train")
def train(CFG: omegaconf.dictconfig.DictConfig):
    run = Run(experiment="drc_centralized_train")
    run['hparams'] = CFG

    if not os.path.exists(CFG.save_path):
        os.makedirs(CFG.save_path)

    CFG = dict(CFG)
    CFG['ann_file'] = CFG['ann_file_train']
    CFG['test_mode'] = False

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(CFG)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(CFG)
    if not CFG.get('cpu', False):
        torch.cuda.set_device(CFG.get('gpu', 0))
        model = model.cuda()
    
    # Build loss
    loss = build_loss(CFG)

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
            for feature, label, _ in dataset:        
                if CFG.get('cpu', False):
                    input, target = feature, label
                else:
                    input, target = feature.cuda(), label.cuda()

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

        epoch += 1
        epoch_loss = 0

if __name__ == "__main__":
    train()
