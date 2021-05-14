# Copyright 2021 Zhongyang Zhang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
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
                 no_augment=True,
                 aug_prob=0.5,
                 batch_size=1):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        
        self.check_files()
        self.count = 0

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
            lr, hr = common.augment(lr, hr, prob=self.aug_prob)
            lr, hr = common.black_square(lr, hr, prob=self.aug_prob)
            if self.count % self.batch_size == 0:
                self.aug_scale = random.choice([1.5, 2, 3, 4])
                self.aug_down_up = 1 if random.random() < self.aug_prob else 0
            lr, hr = common.down_up(
                lr, hr, scales=self.aug_scale, prob=self.aug_down_up, up_prob=0)

        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, color_range=self.color_range)

        self.count += 1

        return lr_tensor, hr_tensor, filename
