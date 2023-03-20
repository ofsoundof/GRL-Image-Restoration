from typing import Optional

import pytorch_lightning as pl
from data.datasets.restoration_bsr import BSRDataset
from data.datasets.restoration_db import (
    DeblurDataset,
)
from data.datasets.restoration_dm import (
    DemosaicDataset,
)
from data.datasets.restoration_dn import DnDataset
from data.datasets.restoration_jpeg import (
    JPEGDataset,
)
from data.datasets.restoration_paired_dataset import (
    PairedDataset,
)
from data.datasets.restoration_sr import SRDataset
from torch.utils.data import DataLoader


class IRDataModule(pl.LightningDataModule):
    """
    Wrapper around the OnDataModule.
    """

    def __init__(self, train, val, test=None, num_train_samples=0, name="", **kwargs):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.num_train_samples = num_train_samples
        self.name = name

    def setup(self, stage: Optional[str] = None):
        if self.name == "dn":
            dataset = DnDataset
        elif self.name == "sr":
            dataset = SRDataset
        elif self.name == "jpeg":
            dataset = JPEGDataset
        elif self.name == "dm":
            dataset = DemosaicDataset
        elif self.name == "db":
            dataset = DeblurDataset
        elif self.name == "bsr":
            dataset = BSRDataset
        elif self.name == "paired":
            dataset = PairedDataset

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = dataset(self.train, "train", self.num_train_samples)
            if self.name == "bsr" and self.val.dataset != "realsr":
                self.val_dataset = SRDataset(self.val, "val")
            else:
                self.val_dataset = dataset(self.val, "val")

        # Assign test dataset for use in dataloader(s)
        elif stage == "validate":
            # self.val_dataset = dataset(self.val, "val")
            if self.name == "bsr" and self.val.dataset != "realsr":
                self.val_dataset = SRDataset(self.val, "val")
            else:
                self.val_dataset = dataset(self.val, "val")
        elif stage == "test":
            self.test_dataset = dataset(self.test, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train.batch_size,
            shuffle=True,
            num_workers=self.train.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.val.batch_size,
            shuffle=False,
            num_workers=self.val.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return None
