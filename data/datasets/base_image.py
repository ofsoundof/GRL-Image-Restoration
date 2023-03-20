import json
import os
import os.path as osp
import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import h5py
import cv2
import numpy as np
import torch.utils.data as data
from omegaconf import DictConfig
from utils.utils_image import modcrop, rgb2ycbcr_np


TRAIN = "train"
VAL = "val"
HOME_DIR = osp.expanduser("~/")
ROOT_DIR = osp.join(HOME_DIR, "projects/data/LightningIR")


DATA_DIR = {
    # cache dir
    "CACHE": osp.join(HOME_DIR, ".on_device_ai/cache"),
    # json file dir
    "JSON": osp.join(ROOT_DIR, "image_info"),
    # test dataset dir
    "TEST": osp.join(ROOT_DIR, "test_set"),
    # training dataset dir
    "DIV2K": osp.join(ROOT_DIR, "DIV2K"),
    "Flickr2K": osp.join(ROOT_DIR, "Flickr2K"),
    "LSDIR": osp.join(ROOT_DIR, "LSDIR"),
    "OST": osp.join(ROOT_DIR, "OST"),
    "SCUT-CTW1500": osp.join(ROOT_DIR, "SCUT-CTW1500"),
    "FFHQ": osp.join(ROOT_DIR, "FFHQ"),
    "BSD400": osp.join(ROOT_DIR, "BSD400"),
    "WED": osp.join(ROOT_DIR, "WED"),
    "imagenet": osp.join(ROOT_DIR, "imagenet"),
    # deblur dataset dir
    "GOPRO": osp.join(ROOT_DIR, "GOPRO"),
    "DPDD": osp.join(ROOT_DIR, "DPDD/dd_dp_dataset_png"),
    "HIDE": osp.join(ROOT_DIR, "HIDE_dataset"),
    "RealBlur": osp.join(ROOT_DIR, "RealBlur"),
}


def load_img_info(dataset, dataset_dir, img_list):
    img_info = []
    for img in img_list:
        out_img = []
        for k, v in img.items():
            if k.find("path") >= 0:
                out_img.append(osp.join(dataset, v))
                out_img.append(osp.join(dataset_dir, v))
        img_info.append(out_img)

    # num_frames = len([k for k, v in img_list[0].items() if k.find("path") >= 0])
    # if num_frames == 1:
    #     img_info = [
    #         (
    #             osp.join(dataset, img["path"]),
    #             osp.join(dataset_dir, img["path"]),
    #         )
    #         for img in img_list
    #     ]
    # elif num_frames == 2:
    #     img_info = [
    #         (
    #             osp.join(dataset, img["path_gt"]),
    #             osp.join(dataset_dir, img["path_gt"]),
    #             osp.join(dataset, img["path_lq"]),
    #             osp.join(dataset_dir, img["path_lq"]),
    #         )
    #         for img in img_list
    #     ]
    # elif num_frames == 3:
    #     img_info = [
    #         (
    #             osp.join(dataset, img["path_gt"]),
    #             osp.join(dataset_dir, img["path_gt"]),
    #             osp.join(dataset, img["path_lq_l"]),
    #             osp.join(dataset_dir, img["path_lq_l"]),
    #             osp.join(dataset, img["path_lq_r"]),
    #             osp.join(dataset_dir, img["path_lq_r"]),
    #         )
    #         for img in img_list
    #     ]

    return img_info


def load_json(path):
    # print("dataset path", osp.join(DATA_DIR["JSON"], path))
    with open(osp.join(DATA_DIR["JSON"], path), "r") as f:
        files = json.load(f)
    return files


def lsdir_mapping(dataset):
    if dataset.find("lsdir_x2") >= 0:
        return "LSDIR_X2"
    elif dataset.find("lsdir_x4") >= 0:
        return "LSDIR_X4"
    elif dataset.find("lsdir_fixed") >= 0:
        return "LSDIR_fixed"
    else:
        return "LSDIR"


def _get_div2k(train: bool, scale: int = 0):
    identifier = "train" if train else "val"
    dataset = "DIV2K"
    filename = f"{identifier}.json" if scale == 0 else f"{identifier}_X{scale}.json"
    img_list = load_json(f"{dataset}/{filename}")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    return img_info


def _get_flickr2k(scale: int = 0):
    dataset = "Flickr2K"
    filename = "train.json" if scale == 0 else f"train_X{scale}.json"
    img_list = load_json(f"{dataset}/{filename}")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    return img_info


def _get_lsdir(dataset: str, split: str = None, scale: int = 0):
    """
    dataset: [set]_[choices]
        set: lsdir, lsdir_x2, lsdir_x4, lsdir_fixed,
        choices: validation - val, val1, val2, val3, val4
                 test - test, test1, test2, test3, test4
                 train - part1, part2, part3, part4, part5, part6, part7, part8, part9
                       - percent**, random_percent**,
    split: train, val, test
    scale: default 0
    """
    if split is None:
        split = "train"
    filename = f"{split}.json" if scale == 0 else f"{split}_X{scale}.json"
    img_list = load_json(f"LSDIR/{filename}")
    img_info = load_img_info(
        lsdir_mapping(dataset),
        DATA_DIR["LSDIR"].replace("LSDIR", lsdir_mapping(dataset)),
        img_list,
    )

    # validation and test choices
    if split in dataset:
        division = int(dataset.split(split)[1][0])
        img_info = img_info[250 * (division - 1) : 250 * division]

    # train, different parts
    if dataset.find("part") >= 0:
        partition_key = f"part{dataset.split('part')[1][0]}_train"
        img_partition_info = load_json("LSDIR/train_image_partition.json")
        img_info_part = []
        for img in img_info:
            if scale == 0:
                if img["path"] in img_partition_info[partition_key]:
                    img_info_part.append(img)
            else:
                if img["path_gt"] in img_partition_info[partition_key]:
                    img_info_part.append(img)
        img_info = img_info_part

    # train, percentage of used images
    if dataset.find("percent") >= 0:
        percent = float(dataset.split("percent")[1]) / 100.0
        if dataset.find("random") >= 0:
            random.shuffle(img_info)
        num_imgs = int(len(img_info) * percent)
        img_info = img_info[:num_imgs]

    return img_info


def _get_imagenet(split=None):
    # split: train, test, val
    if split is None:
        split = "train"
    dataset = "imagenet"
    img_list = load_json(f"{dataset}/{split}.json")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    return img_info


def _get_ost() -> List[Tuple[str, str]]:
    dataset = "OST"
    img_list = load_json(f"{dataset}/train.json")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    assert len(img_info) == 10324
    return img_info


def _get_scut_ctw1500() -> List[Tuple[str, str]]:
    dataset = "SCUT-CTW1500"
    img_list = load_json(f"{dataset}/train.json")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    assert len(img_info) == 1500
    return img_info
    # return img_info[:num_imgs]


def _get_ffhq() -> List[Tuple[str, str]]:
    dataset = "FFHQ"
    img_list = load_json(f"{dataset}/train.json")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    assert len(img_info) == 10000
    return img_info


def _get_bsd400() -> List[Tuple[str, str]]:
    dataset = "BSD400"
    img_list = load_json(f"{dataset}/train.json")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    assert len(img_info) == 400
    return img_info


def _get_wed() -> List[Tuple[str, str]]:
    dataset = "WED"
    img_list = load_json(f"{dataset}/train.json")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    assert len(img_info) == 4744
    return img_info


def imread(path, config):
    num_channels = config.num_channels
    # print("read image from", path)
    if num_channels == 1:
        if "quality_factor" in config and config.dataset in [
            "live1",
            "bsds500",
            "urban100",
        ]:
            # image = imageio.imread(f, as_gray=False, pilmode="RGB").astype(np.uint8)
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            image = rgb2ycbcr_np(image, y_only=True)
            image = np.expand_dims(image, axis=2)
        else:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # image = imageio.imread(f, as_gray=True).astype(np.uint8)
            image = np.expand_dims(image, axis=2)
    else:
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        # image = imageio.imread(f, as_gray=False, pilmode="RGB").astype(np.uint8)
    return image


def _random_index(img, patch_size):
    h, w = img.shape[:2]
    x = random.randrange(0, h - patch_size + 1)
    y = random.randrange(0, w - patch_size + 1)
    return x, y


# def _sample_patches(imgs, x, y, patch_size, scale):
#     # pad the image if necessary
#     h, w = imgs[0].shape[:2]
#     if h < patch_size * scale or w < patch_size * scale:
#         h_pad = max(0, patch_size * scale - h)
#         w_pad = max(0, patch_size * scale - w)
#         padding = ((0, h_pad), (0, w_pad), (0, 0))
#         imgs = [np.pad(img, padding, "constant", constant_values=0) for img in imgs]

#     # sample patch while training
#     x0 = x * scale
#     y0 = y * scale
#     x1 = x0 + patch_size * scale
#     y1 = y0 + patch_size * scale
#     return [img[x0:x1, y0:y1] for img in imgs]


def _pad_images(imgs, patch_size, scale):
    # pad the image if necessary
    h, w = imgs[0].shape[:2]
    if h < patch_size * scale or w < patch_size * scale:
        h_pad = max(0, patch_size * scale - h)
        w_pad = max(0, patch_size * scale - w)
        padding = ((0, h_pad), (0, w_pad), (0, 0))
        imgs = [np.pad(img, padding, "constant", constant_values=0) for img in imgs]
    return imgs


def _sample_patches(imgs, x, y, patch_size, scale):
    # sample patch while training
    x0 = x * scale
    y0 = y * scale
    x1 = x0 + patch_size * scale
    y1 = y0 + patch_size * scale
    return [img[x0:x1, y0:y1] for img in imgs]


class ImageBaseDataset(ABC, data.Dataset):
    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.num_train_samples = num_train_samples
        if num_train_samples != 0:
            self.train_size = len(self.img_info)

        self.use_hdf5 = True if self.stage == TRAIN else False

        # dataset = self.cfg.dataset.lower()
        # # if (
        # #     dataset.find("lsdir") >= 0
        # #     and dataset.find("x4") < 0
        # #     and dataset.find("x2") < 0
        # # ):
        # if dataset == "lsdir":
        #     self.use_hdf5 = False

    def _get_index(self, index: int) -> int:
        if self.stage == TRAIN:
            if self.num_train_samples == 0:
                index = index // self.cfg.num_patches
            else:
                index = index % self.train_size
        return index

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    def _load_item(self, index: int) -> np.ndarray:
        img_gt = self.img_info[index]
        img_gt = self._cache_image(img_gt)
        return img_gt

    def _cache_image(self, path: Tuple[str, str]) -> np.ndarray:
        image = path[1]
        if self.use_hdf5:
            cache = osp.join(osp.expanduser(DATA_DIR["CACHE"]), path[0])
            if not osp.exists(cache):
                image = imread(image, self.cfg)
                with h5py.File(cache + ".cache_tmp", "w", libver="latest") as f:
                    f.create_dataset(
                        "image",
                        data=image,
                        maxshape=image.shape,
                        compression="lzf",
                        shuffle=True,
                        track_times=False,
                        track_order=False,
                    )
                os.replace(cache + ".cache_tmp", cache)
            return h5py.File(cache, "r", libver="latest")["image"][()]
        else:
            # print("do not use hdf5")
            image = imread(image, self.cfg)
            return image

    def _augment(self, images):
        if not isinstance(images, list):
            images = [images]

        if self.stage == TRAIN:
            # augmentation while training
            # random flip: 8 combinations in total
            if random.random() < 0.5:
                images = [x[::-1] for x in images]
            if random.random() < 0.5:
                images = [x[:, ::-1] for x in images]
            if random.random() < 0.5:
                images = [np.swapaxes(x, 0, 1) for x in images]

        if len(images) == 1:
            images = images[0]
        return images

    def __len__(self) -> int:
        if self.stage == TRAIN:
            if self.num_train_samples == 0:
                return len(self.img_info) * self.cfg.num_patches
            else:
                return self.num_train_samples
        else:
            return len(self.img_info)

    def _sample_patch(
        self,
        imgs_H: Tuple,
        imgs_L: Tuple = None,
        scale: int = 1,
        # modulo: int = 1,
    ) -> Tuple:

        if imgs_L is not None:
            if not isinstance(imgs_L, (list, tuple)):
                imgs_L = [imgs_L]
            if not isinstance(imgs_H, (list, tuple)):
                imgs_H = [imgs_H]

            if self.stage == TRAIN:
                imgs_L = _pad_images(imgs_L, self.patch_size, 1)
                imgs_H = _pad_images(imgs_H, self.patch_size, scale)
                x, y = _random_index(imgs_L[0], self.patch_size)
                imgs_L = _sample_patches(imgs_L, x, y, self.patch_size, 1)
                imgs_H = _sample_patches(imgs_H, x, y, self.patch_size, scale)
            else:
                imgs_H = [modcrop(img, scale) for img in imgs_H]

            if len(imgs_L) == 1:
                imgs_L = imgs_L[0]
            if len(imgs_H) == 1:
                imgs_H = imgs_H[0]
            return imgs_H, imgs_L

        else:
            if not isinstance(imgs_H, list):
                imgs_H = [imgs_H]

            if self.stage == TRAIN:
                imgs_H = _pad_images(imgs_H, self.patch_size, 1)
                x, y = _random_index(imgs_H[0], self.patch_size)
                imgs_H = _sample_patches(imgs_H, x, y, self.patch_size, 1)
            else:
                modulo = self.cfg.get("modulo", 8)
                # TODO: whether need to be multiples of 8?
                x = imgs_H[0].shape[0] // modulo * modulo
                y = imgs_H[0].shape[1] // modulo * modulo
                imgs_H = [img[:x, :y] for img in imgs_H]

            if len(imgs_H) == 1:
                imgs_H = imgs_H[0]
            return imgs_H
