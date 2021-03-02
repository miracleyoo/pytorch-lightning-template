import os.path as op
import numpy as np
import pickle as pkl
from pathlib2 import Path

import torch.utils.data as data
from . import common


class SatupData(data.Dataset):
    def __init__(self, data_dir='dataset',
                 color_range=255,
                 train=True,
                 augment=True, **kwargs):
        self.data_dir = data_dir
        self.train = train
        self.color_range = color_range
        self.check_files()
        self.aug = augment and train

    def check_files(self):
        middir = 'train' if self.train else 'val'
        info_file = Path(self.data_dir, f'{middir}_lr.pkl')
        with open(info_file, 'rb') as f:
            self.lr_list = pkl.load(f)

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, idx):
        lrfile = self.lr_list[idx]
        hrfile = self.lr_list[idx].replace('LRBigEarth', 'SRBigEarth')
        filename = op.splitext(op.basename(hrfile))[0]
        lr = np.load(lrfile).transpose(1, 2, 0)
        hr = np.load(hrfile).transpose(1, 2, 0)
        lr = common.bitdepth_convert(lr, src=16, dst=8)
        hr = common.bitdepth_convert(hr, src=16, dst=8)

        if self.aug:
            lr, hr = common.augment(lr, hr)
            lr, hr = common.black_square(lr, hr)

        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, color_range=self.color_range)

        return lr_tensor, hr_tensor, filename
