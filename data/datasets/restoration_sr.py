from typing import List, Tuple

import cv2

import numpy as np
import torchvision.transforms as transforms
from omegaconf import DictConfig
from data.datasets.base_image import (
    _get_div2k,
    _get_flickr2k,
    _get_imagenet,
    _get_lsdir,
    DATA_DIR,
    ImageBaseDataset,
    load_img_info,
    load_json,
    TRAIN,
    VAL,
)
from utils.matlab_functions import imresize
from utils.utils_bsr.utils_image import (
    single2uint,
    uint2single,
)
from utils.utils_bsr.utils_usm import usm_sharp
from utils.utils_image import modcrop


def get_train_file(dataset: str, scale: str) -> List[Tuple[str, str]]:
    dataset = dataset.lower()
    # DIV2K and Flickr2K
    if dataset == "div2k" or dataset.find("df2k") >= 0:
        # div2k: DIV2K 800 training images
        # df2k: DIV2K 800 + Flickr2K 2650
        # df2k3350: DIV2K 800 + DIV2K validation 100 + Flickr2K 2650
        img_info = _get_div2k(True, scale)
        if dataset.find("df2k") >= 0:
            img_info += _get_flickr2k(scale)
        if dataset.find("3550") >= 0:
            img_info += _get_div2k(False, scale)
    # LSDIR
    elif dataset.find("lsdir") >= 0:
        img_info = _get_lsdir(dataset, "train", scale)
        if dataset.find("extended") >= 0:
            img_info += _get_div2k(True, scale) + _get_flickr2k(scale)
    # ImageNet
    elif dataset.find("imagenet") >= 0:
        img_info = _get_imagenet()
    else:
        raise NotImplementedError(f"SISR training dataset {dataset} not implemented.")
    print(f"Training dataset {dataset} with {len(img_info)} images.")
    return img_info


def get_test_file(dataset: str, scale: int) -> List[Tuple[str, str]]:
    dataset = dataset.lower()

    dataset_mapping = {
        "set5": "Set5",
        "set14": "Set14",
        "bsd100": "B100",
        "b100": "B100",
        "urban100": "Urban100",
        "manga109": "Manga109",
    }

    if dataset.find("div2k") >= 0:
        img_info = _get_div2k(False, scale)
    elif dataset.find("lsdir") >= 0:
        if dataset.find("val") >= 0:
            img_info = _get_lsdir(dataset, "val", scale)
        elif dataset.find("test") >= 0:
            img_info = _get_lsdir(dataset, "test", scale)
    elif dataset in dataset_mapping:
        test_set = dataset_mapping[dataset]
        img_list = load_json(f"{test_set}/test_X{scale}.json")
        img_info = load_img_info(test_set, DATA_DIR["TEST"], img_list)
    else:
        raise NotImplementedError(f"SISR test dataset {dataset} not implemented.")
    print(f"Validation set {dataset} with {len(img_info)} images.")
    return img_info


class SRDataset(ImageBaseDataset):
    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:
        self.scale = cfg.scale
        self.load_lr = cfg.get("load_lr", False)
        if stage == TRAIN:
            self.patch_size = cfg.lr_patch_size  # lr patch size
            self.img_info = get_train_file(cfg.dataset, cfg.scale)
        else:
            self.img_info = get_test_file(cfg.dataset, cfg.scale)
            # for blind image sr
            self.use_usm = cfg.get("use_usm", False)
        super(SRDataset, self).__init__(cfg, stage, num_train_samples)

    def __getitem__(self, index: int):

        # load image and sample a patch
        index = self._get_index(index)
        img_lq, img_gt = self._load_item(index)
        img_gt, img_lq = self._sample_patch(img_gt, img_lq, self.scale)
        img_lq, img_gt = self._augment([img_lq, img_gt])

        if self.stage == VAL:
            if self.use_usm:
                img_gt = uint2single(img_gt)
                img_gt = usm_sharp(img_gt)
                img_gt = single2uint(img_gt)

        img_lq = np.ascontiguousarray(img_lq)
        img_gt = np.ascontiguousarray(img_gt)

        img_lq = transforms.functional.to_tensor(img_lq)
        img_gt = transforms.functional.to_tensor(img_gt)

        # indices are used by torchmetrics
        return {
            "indices": index,
            "img_lq": img_lq,
            "img_gt": img_gt,
            "filenames": self.img_info[index][0],
        }

    def _load_item(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img_gt = self._cache_image(self.img_info[index][0:2])
        if self.stage == VAL or self.load_lr:
            img_lq = self._cache_image(self.img_info[index][2:])
        else:
            img_gt = modcrop(img_gt, self.scale)
            # img_lq = imresize_np(img_gt, 1 / self.scale)
            # img_lq = np.clip(img_lq, 0, 255).astype(np.uint8)

            h, w = img_gt.shape[:2]
            h = max(h, self.patch_size * self.scale)
            w = max(w, self.patch_size * self.scale)
            img_gt = cv2.resize(cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR), (w, h))
            img_lq = cv2.cvtColor(
                imresize(img_gt / 255.0, 1 / self.scale), cv2.COLOR_BGR2RGB
            )
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        return img_lq, img_gt

    # def _sample_patch(
    #     self, img_lq: np.ndarray, img_gt: np.ndarray
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     if self.stage == TRAIN:
    #         # pad the image if necessary
    #         h, w, c = img_lq.shape
    #         if h < self.lr_ps or w < self.lr_ps:
    #             h_pad = max(0, self.lr_ps - h)
    #             w_pad = max(0, self.lr_ps - w)
    #             img_pad_lq = ((0, h_pad), (0, w_pad), (0, 0))
    #             img_pad_gt = ((0, h_pad * self.scale), (0, w_pad * self.scale), (0, 0))

    #             img_lq = np.pad(img_lq, img_pad_lq, "constant", constant_values=0)
    #             img_gt = np.pad(img_gt, img_pad_gt, "constant", constant_values=0)

    #         # sample patch while training
    #         x = random.randrange(0, img_lq.shape[0] - self.lr_ps + 1)
    #         y = random.randrange(0, img_lq.shape[1] - self.lr_ps + 1)

    #         img_lq = img_lq[x : x + self.lr_ps, y : y + self.lr_ps]
    #         img_gt = img_gt[
    #             x * self.scale : (x + self.lr_ps) * self.scale,
    #             y * self.scale : (y + self.lr_ps) * self.scale,
    #         ]

    #     else:
    #         img_gt = modcrop(img_gt, self.scale)
    #         # sz = img_gt.shape
    #         # img_lq = img_lq[: sz[0] // self.scale, : sz[1] // self.scale, :]
    #     return img_lq, img_gt
