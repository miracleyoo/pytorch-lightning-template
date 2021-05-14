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

import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

from torchvision import transforms
from sklearn.model_selection import train_test_split


class StandardData(data.Dataset):
    def __init__(self, data_dir=r'data/ref',
                 class_num=9,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment

        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        file_list_path = op.join(self.data_dir, 'file_list.pkl')
        with open(file_list_path, 'rb') as f:
            file_list = pkl.load(f)

        fl_train, fl_val = train_test_split(
            file_list, test_size=0.2, random_state=2333)
        self.path_list = fl_train if self.train else fl_val

        label_file = './data/ref/label_dict.pkl'
        with open(label_file, 'rb') as f:
            self.label_dict = pkl.load(f)

    def __len__(self):
        return len(self.path_list)

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out

    def __getitem__(self, idx):
        path = self.path_list[idx]
        filename = op.splitext(op.basename(path))[0]
        img = np.load(path).transpose(1, 2, 0)

        labels = self.to_one_hot(self.label_dict[filename.split('_')[0]])
        labels = torch.from_numpy(labels).float()

        trans = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
            transforms.RandomCrop(128),
            transforms.Normalize(self.img_mean, self.img_std)
        ) if self.train else torch.nn.Sequential(
            transforms.CenterCrop(128),
            transforms.Normalize(self.img_mean, self.img_std)
        )

        img_tensor = trans(img)

        return img_tensor, labels, filename