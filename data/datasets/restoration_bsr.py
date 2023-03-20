import os
import random
from typing import Dict, List, Tuple

import cv2
import torchvision.transforms as T
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
from utils.utils_bsr import (
    utils_image as util,
    utils_isp,
    utils_sisr as sr,
)
from utils.utils_bsr.utils_usm import usm_sharp


def get_train_file_bsr(dataset: str) -> List[Tuple[str, str]]:
    dataset = dataset.lower()

    if dataset == "all":
        img_info = (
            get_train_file("lsdir_x4_extended")
            + get_train_file("ost")
            + get_train_file("scut_ctw1500")[200:]
            + get_train_file("ffhq")
        )
    else:
        img_info = get_train_file(dataset)
    return img_info


class BSRDataset(ImageBaseDataset):
    """
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    """

    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:

        self.sf = cfg.get("scale", 4)
        self.n_channels = cfg.get("num_channels", 3)
        if stage == TRAIN:
            self.patch_size = cfg.lr_patch_size  # lr patch size
            use_usm_pixel = cfg.get("use_usm_pixel", False)
            use_usm_percep = cfg.get("use_usm_percep", False)
            use_usm_gan = cfg.get("use_usm_gan", False)
            self.use_usm = use_usm_pixel or use_usm_percep or use_usm_gan

            self.img_info = get_train_file_bsr(cfg.dataset)
        else:
            self.img_info = get_test_file(cfg.dataset)

        self.ispmodel = utils_isp.ISPModel()
        self.jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        super(BSRDataset, self).__init__(cfg, stage, num_train_samples)

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        index = self._get_index(index)
        img_gt = self._load_item(index)
        img_gt = self._augment(img_gt)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.stage == TRAIN:

            H, W, C = img_gt.shape
            crop_pad_size = 400
            # pad
            if H < crop_pad_size or W < crop_pad_size:
                pad_h = max(0, crop_pad_size - H)
                pad_w = max(0, crop_pad_size - W)
                img_gt = cv2.copyMakeBorder(
                    img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
                )
            # crop
            H, W, C = img_gt.shape
            x = random.randint(0, max(0, H - crop_pad_size))
            y = random.randint(0, max(0, W - crop_pad_size))
            img_gt = img_gt[x : x + crop_pad_size, y : y + crop_pad_size, :]

            img_gt = self.jitter(util.uint2tensor4(img_gt))
            img_gt = util.tensor2single(img_gt)

            # USM
            img_gt_usm = usm_sharp(img_gt) if self.use_usm else img_gt
            # print(self.img_info[index][0])
            # img_gt = util.uint2single(img_gt)
            img_lq, img_gt_usm = sr.degradation_sr2(img_gt_usm, self.sf, self.ispmodel)
            (img_gt, img_gt_usm), img_lq = self._sample_patch(
                [img_gt, img_gt_usm], img_lq, self.sf
            )
        else:
            img_gt = util.uint2single(img_gt)
            if self.cfg.with_gt:
                img_lq, img_gt = sr.degradation_sr2(img_gt, self.sf, self.ispmodel)
                img_gt_usm = None
            else:
                img_lq = img_gt
                img_gt, img_gt_usm = 0, None
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        if not (isinstance(img_gt, int) and img_gt == 0):
            img_gt = util.single2tensor3(img_gt)
        if img_gt_usm is not None:
            img_gt_usm = util.single2tensor3(img_gt_usm)
        img_lq = util.single2tensor3(img_lq)

        data = {
            "indices": index,
            "img_lq": img_lq,
            "img_gt": img_gt,
            "filenames": self.img_info[index][0],
        }
        if img_gt_usm is not None:
            data["img_gt_usm"] = img_gt_usm

        return data

