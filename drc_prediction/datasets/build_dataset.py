# Copyright 2022 CircuitNet. All rights reserved.

from torch.utils.data import DataLoader
from datasets.drc_dataset import DRCDataset
import time


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self


def build_dataset(opt):
    dataset = DRCDataset(**opt)
    if opt['test_mode']:
        return DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)
    else:
        return IterLoader(DataLoader(dataset=dataset, num_workers=16, batch_size=opt.pop('batch_size'), shuffle=True, drop_last=True, pin_memory=True))
