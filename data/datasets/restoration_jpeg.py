import random
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from data.datasets.base_image import (
    ImageBaseDataset,
    TRAIN,
    VAL,
)
from data.datasets.restoration_dn import (
    get_test_file,
    get_train_file,
)


class JPEGDataset(ImageBaseDataset):
    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:
        if stage == TRAIN:
            self.patch_size = cfg.patch_size
            self.quality_factor_range = cfg.quality_factor_range
            self.img_info = get_train_file(cfg.dataset)  # [:800]
        else:
            self.img_info = get_test_file(cfg.dataset)
        self.quality_factor = cfg.quality_factor
        super(JPEGDataset, self).__init__(cfg, stage, num_train_samples)

    def __getitem__(self, index: int):

        # load image and sample a patch
        index = self._get_index(index)
        img_gt = self._load_item(index)
        if self.stage == TRAIN and self.cfg.get("patchwise", False):
            img_gt = self._sample_patch(img_gt)
            img_gt = self._augment(img_gt)
            img_lq, quality_factor = self.jpeg_compress(img_gt)
        else:
            img_lq, quality_factor = self.jpeg_compress(img_gt)
            img_gt, img_lq = self._sample_patch(img_gt, img_lq)
            img_lq, img_gt = self._augment([img_lq, img_gt])

        img_lq = np.ascontiguousarray(img_lq)
        img_gt = np.ascontiguousarray(img_gt)
        img_lq = transforms.functional.to_tensor(img_lq)
        img_gt = transforms.functional.to_tensor(img_gt)

        if self.cfg.noise_level_map:
            noise_level_map = torch.ones((1, *img_lq.shape[1:]))
            noise_level_map = noise_level_map.mul_(1 - quality_factor / 100).float()
            img_lq = torch.cat((img_lq, noise_level_map), 0)

        return {
            "indices": index,
            "img_lq": img_lq,
            "img_gt": img_gt,
            "filenames": self.img_info[index][0],
        }

    def jpeg_compress(self, img_gt: np.ndarray) -> Tuple[np.ndarray, int]:
        quality_factor = self.quality_factor

        if self.stage == TRAIN and len(self.quality_factor_range) > 0:
            quality_factor = random.randint(*self.quality_factor_range)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
        if self.cfg.num_channels == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
            result, encimg = cv2.imencode(".jpg", img_gt, encode_params)
            img_lq = cv2.imdecode(encimg, 1)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
        else:
            result, encimg = cv2.imencode(".jpg", img_gt, encode_params)
            img_lq = cv2.imdecode(encimg, 0)
            img_lq = img_lq[..., np.newaxis]

        return img_lq, quality_factor

    # def _sample_patch(self, imgs: Tuple[np.ndarray]) -> Tuple[np.ndarray]:
    #     if self.stage == TRAIN:
    #         # pad the image is necessary
    #         h, w, c = imgs[0].shape
    #         if h < self.patch_size or w < self.patch_size:
    #             h_pad = max(0, self.patch_size - h)
    #             w_pad = max(0, self.patch_size - w)
    #             img_pad = ((0, h_pad), (0, w_pad), (0, 0))
    #             imgs = [
    #                 np.pad(img, img_pad, "constant", constant_values=0) for img in imgs
    #             ]

    #         # sample patch while training
    #         x = random.randrange(0, imgs[0].shape[0] - self.patch_size + 1)
    #         y = random.randrange(0, imgs[0].shape[1] - self.patch_size + 1)
    #         imgs = [
    #             img[x : x + self.patch_size, y : y + self.patch_size] for img in imgs
    #         ]
    #     else:
    #         x = imgs[0].shape[0] // 8 * 8  # TODO: whether need to be multiples of 8?
    #         y = imgs[0].shape[1] // 8 * 8
    #         imgs = [img[:x, :y] for img in imgs]
    #     return imgs
