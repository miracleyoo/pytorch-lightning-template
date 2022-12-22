# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='mpsnr',
        mode='max',
        patience=10,
        min_delta=0.01
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='mpsnr',
        filename='best-{epoch:02d}-{mpsnr:.2f}-{mssim:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.ckpt_path = load_path

    args.callbacks = load_callbacks()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)



if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='satup_data', type=str)
    parser.add_argument('--data_dir', default='dataset', type=str)
    parser.add_argument('--model_name', default='rdn_fuse', type=str)
    parser.add_argument('--loss', default='l1', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')

    # Model Hyperparameters
    parser.add_argument('--scale', default=2, type=int)
    parser.add_argument('--in_bands_num', default=12, type=int)
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--rdn_size', default=3, type=int)
    parser.add_argument('--rdb_growrate', default=64, type=int)
    parser.add_argument('--rdb_conv_num', default=8, type=int)

    # Other
    parser.add_argument('--color_range', default=255, type=int)
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=250)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [1.315, 1.211, 1.948, 1.892, 3.311,
                     6.535, 7.634, 8.197, 8.395, 8.341, 5.89, 3.616]
    args.std_sen = [5.958, 2.273, 2.299, 2.668, 2.895,
                    4.276, 4.978, 5.237, 5.304, 5.103, 4.298, 3.3]

    main(args)
