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
import cv2
import torch
import numpy as np
import skimage.color as sc
import pickle as pkl

from operator import itemgetter


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def even_sample(items, number):
    """ Evenly sample `number` items from `items`.
    """
    indexs = (np.linspace(0, len(items)-1, number).astype(int)).tolist()
    return itemgetter(*indexs)(items)


def get_patch(*args, patch_size=96, scale=1, its=None):
    """ Every image has a different aspect ratio. In order to make 
        the input shape the same, here we crop a 96*96 patch on LR 
        image, and crop a corresponding area(96*r, 96*r) on HR image.
    Args:
        args: lr, hr
        patch_size: The x and y length of the crop area on lr image.
        scale: r, upscale ratio
    Returns:
        0: cropped lr image.
        1: cropped hr image.
    """
    ih, iw = args[0].shape[:2]

    tp = int(scale * patch_size)
    ip = patch_size

    ix = random.randrange(0, (iw-ip))
    iy = random.randrange(0, (ih-ip))

    tx, ty = int(scale * ix), int(scale * iy)

    if its is None:
        its = np.zeros(len(args)-1, int)
    itp = (tp, ip)
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[(ty, iy)[b]:(ty, iy)[b] + itp[b], (tx, ix)[b]:(tx, ix)[b] + itp[b], :] for a, b in zip(args[1:], its)]
    ]
    return ret


def set_channel(*args, n_channels=3):
    """ Do the channel number check. If input channel is 
        not n_channels, convert it to n_channels.
    Args:
        n_channels: the target channel number.
    """
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def bitdepth_convert(image, src=16, dst=8):
    """ Convert images with different bit depth.
    Args:
        image: Input image, and ndarry.
        src: source bit depth.
        dst: target bit depth.
    """
    coe = src - dst
    image = (image + 1) / (2 ** coe) - 1
    return image


def np2Tensor(*args, color_range=255):
    """ Transform an numpy array to tensor. Each single value in
        the target tensor will be mapped into [0,1]
    Args:
        color_range: Max value of a single pixel in the original array.
    """
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(color_range / 255)
        return tensor

    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True, prob=0.5):
    """ Same data augmentation for a series of input images.
        Operations included: random horizontal flip, random vertical 
        flip, random 90 degree rotation.
    Args:
        args: A list/tuple of images.
        hflip: Whether use random horizontal flip.
        rot: Whether use random vertical flip and rotation.
        prob: The probobility of using augment.
    """
    hflip = hflip and random.random() < prob
    vflip = rot and random.random() < prob
    rot90 = rot and random.random() < prob

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]


def black_square(lr, hr, prob=0.5):
    """ Randomly select an edge of square between `min_edge//8` to
        `min_edge//4` and put the square to a random position.
    Args:
        lr: LR image.
        hr: HR image.
        scale: HR/LR scale.
        prob: Probability of adding this black square. 
    """
    if random.random() < prob:
        h, w = lr.shape[:2]
        scale = hr.shape[0] // h
        max_edge = min(h, w)//4
        min_edge = min(h, w)//8
        edge = random.choice(range(min_edge, max_edge))

        start_y = random.choice(range(h-edge))
        start_x = random.choice(range(w-edge))

        lr[start_y:(start_y+edge), start_x:(start_x+edge), :] = 0
        hr[int(start_y*scale):int((start_y+edge)*scale),
           int(start_x * scale):int((start_x+edge)*scale), :] = 0
    return lr, hr


def down_up(*args, scales, up_prob=1, prob=0.5):
    """ Downscale and then upscale the input iamges.
    Args:
        args: A list/tuple of images.
        scales: donw-up scale list. Like: (1.5, 2, 3, 4)
        up_prob: Probability of applying upsample after downsample.
        prob: Probability of applying this augment. 
    """
    def _down_up(img):
        if decision:
            img = cv2.resize(img, None, fx=1/scale, fy=1/scale,
                             interpolation=cv2.INTER_CUBIC)  # INTER_AREA
            if up:
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
        return img

    decision = random.random() < prob
    up = random.random() < up_prob
    scale = random.choice(scales) if type(scales) in (list, tuple) else scales
    return [_down_up(a) for a in args]
