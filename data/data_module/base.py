from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def image_batch_collate_fn(batch):
    """
    convert individual samples loaded to a batched form
        - used in pytorch's DataLoader
    Outputs:
        batch: {
            "x": noisy_img,
            "y": clean_img
            "shot_noise": shot_noise
            "read_noise": read_noise
            "path_y": img path
        }
    """
    batch_x = torch.stack([b[0] for b in batch])
    batch_y = torch.stack([b[1] for b in batch])
    batch_shot_noise = torch.stack([b[2] for b in batch])
    batch_read_noise = torch.stack([b[3] for b in batch])
    batch_path_y = [b[4] for b in batch]
    return {
        "x": batch_x,
        "y": batch_y,
        "shot_noise": batch_shot_noise,
        "read_noise": batch_read_noise,
        "path_y": batch_path_y,
    }


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test=None, **kwargs):
        super().__init__()
        self.train, self.val, self.test = train, val, test
        # self.collate_fn = image_batch_collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train.batch_size,
            persistent_workers=True,
            num_workers=self.train.num_workers,
            # collate_fn=self.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.train.num_workers,
            persistent_workers=True,
            batch_size=self.val.batch_size,
            # collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        if self.test is not None:
            return DataLoader(
                self.test_dataset,
                num_workers=self.train.num_workers,
                persistent_workers=True,
                batch_size=self.test.batch_size,
                # collate_fn=self.collate_fn,
            )
        return None
