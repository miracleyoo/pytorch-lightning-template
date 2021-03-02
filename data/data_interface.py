import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from importlib import import_module


class DInterface(pl.LightningDataModule):

    def __init__(self, batch_size=64,
                 num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.load_data_module()

    def load_data_module(self):
        if self.dataset == 'recursive_up':
            self.data_module = getattr(import_module(
                'data.recursive_up'), 'RecursiveUp')
        elif self.dataset == 'satup_data':
            self.data_module = getattr(
                import_module('data.satup_data'), 'SatupData')
        else:
            raise ValueError('Invalid Dataset Type!')

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.data_module(train=True, **self.kwargs)
            self.valset = self.data_module(train=False, **self.kwargs)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.data_module(train=False, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)