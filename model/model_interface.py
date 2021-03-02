import numpy as np
import torch
import importlib
from torch import nn

import pytorch_lightning as pl
from .metrics import tensor_accessment
from .utils import quantize

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        
        # Project-Specific Definitions
        self.hsi_index = np.r_[0,4:12]
        self.rgb_index = (3,2,1)

    def load_model(self):
        if self.hparams.model_name == 'rdnfuse':
            Model = getattr(importlib.import_module('model.rdnfuse'), 'RDNFuse')
            self.model = Model(self.hparams.scale,
                               self.hparams.in_bands_num,
                               self.hparams.hid,
                               self.hparams.block_num,
                               self.hparams.rdn_size,
                               self.hparams.rdb_growrate,
                               self.hparams.rdb_conv_num,
                               self.hparams.mean_sen,
                               self.hparams.std_sen)
        else:
            raise ValueError('Invalid Module Name!')

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = nn.MSELoss()
        elif loss == 'l1':
            self.loss_function = nn.L1Loss()
        else:
            raise ValueError("Invalid Loss Type!")

    def forward(self, lr_hsi, hr_rgb):
        return self.model(lr_hsi, hr_rgb)

    def training_step(self, batch, batch_idx):
        lr, hr, _ = batch
        sr = self(lr, hr[:,self.rgb_index,])
        loss = self.loss_function(sr[:,self.hsi_index], hr[:,self.hsi_index])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, _ = batch
        sr = self(lr, hr[:,self.rgb_index,])
        sr = quantize(sr, self.hparams.color_range)
        mpsnr, mssim, _, _ = tensor_accessment(
            x_pred=sr[:,self.hsi_index].cpu().numpy(), 
            x_true=hr[:,self.hsi_index].cpu().numpy(), 
            data_range=self.hparams.color_range, 
            multi_dimension=False)

        loss = self.loss_function(sr, hr)
        self.log('mpsnr', mpsnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mssim', mssim, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
