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

import numpy as np
from pathlib2 import Path

import torch.utils.data as data
from . import common


class RecursiveUp(data.Dataset):
    def __init__(self, data_dir='dataset',
                 train=True):
        super().__init__()
        self.train = train
        self.root = Path(data_dir)
        self.check_data()

    def check_data(self):
        self.sentinel_root = self.root / 'sentinel'
        self.drone_root = self.root / 'drone'
        self.filelist = [f.name for f in self.sentinel_root.iterdir()]
        self.filelist = self.filelist[:-1] if self.train else self.filelist[-1:]
        self.scale_strs = [str(2**i)+'x' for i in range(9)]

    def __getitem__(self, idx):
        sen = np.load(self.sentinel_root / self.filelist[idx])
        drone = [np.load(self.drone_root / scale_str / self.filelist[idx])
                 for scale_str in self.scale_strs]
        sen = sen.transpose(1, 2, 0)
        drone = [d.transpose(1, 2, 0) for d in drone]
        sen = common.np2Tensor(sen)[0]
        drone = common.np2Tensor(*drone)
        return sen, drone

    def __len__(self):
        return len(self.filelist)
