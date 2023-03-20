from hashlib import sha256
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from data.datasets.base_image import (
    _get_bsd400,
    _get_div2k,
    _get_ffhq,
    _get_flickr2k,
    _get_imagenet,
    _get_lsdir,
    _get_ost,
    _get_scut_ctw1500,
    _get_wed,
    DATA_DIR,
    ImageBaseDataset,
    load_img_info,
    load_json,
    TRAIN,
    VAL,
)


def _get_add_train() -> List[Tuple[str, str]]:
    return _get_flickr2k() + _get_bsd400() + _get_wed()


def get_train_file(dataset: str) -> List[Tuple[str, str]]:
    dataset = dataset.lower()
    # DIV2K and Flickr2K
    if dataset == "div2k" or dataset.find("df2k") >= 0:
        # div2k: DIV2K 800 training images
        # df2k: DIV2K 800 + Flickr2K 2650
        # df2k3350: DIV2K 800 + DIV2K validation 100 + Flickr2K 2650
        # div2k_extended: DIV2K 800 + Flickr2K 2650 + BSD400 + WED
        img_info = _get_div2k(True)
        if dataset.find("df2k") >= 0:
            img_info += _get_flickr2k()
        if dataset.find("3550") >= 0:
            img_info += _get_div2k(False)
        if dataset == "div2k_extended":
            img_info += _get_add_train()
    # LSDIR
    elif dataset.find("lsdir") >= 0:
        img_info = _get_lsdir(dataset, "train")
        if dataset.find("extended") >= 0:
            img_info += _get_div2k(True) + _get_add_train()
    # ImageNet
    elif dataset.find("imagenet") >= 0:
        img_info = _get_imagenet()
    elif dataset == "ost":
        img_info = _get_ost()
    elif dataset == "scut_ctw1500":
        img_info = _get_scut_ctw1500()
    elif dataset == "ffhq":
        img_info = _get_ffhq()
    else:
        raise NotImplementedError(f"Image denoising dataset {dataset} not implemented.")
    print(f"Training dataset {dataset} with {len(img_info)} images.")
    return img_info


def get_test_file(dataset: str) -> List[Tuple[str, str]]:
    dataset = dataset.lower()

    dataset_mapping = {
        # Denoising
        "set12": "Set12",
        "bsd68": "BSD68",
        "cbsd68": "CBSD68",
        "kodak24": "Kodak24",
        "mcmaster": "McMaster",
        "urban100": "Urban100",
        # JPEG
        "classic5": "Classic5",
        "live1": "LIVE1",
        "bsds500": "BSDS500",
        "icb_gray": "ICB_Gray",
        "icb_rgb": "ICB_RGB",
        # Real SR
        "realsr": "RealSRSetPlus5images",
    }

    if dataset.find("div2k") >= 0:
        img_info = _get_div2k(False)
    elif dataset.find("lsdir") >= 0:
        if dataset.find("val") >= 0:
            img_info = _get_lsdir(dataset, "val")
        elif dataset.find("test") >= 0:
            img_info = _get_lsdir(dataset, "test")
    elif dataset in dataset_mapping:
        test_set = dataset_mapping[dataset]
        img_list = load_json(f"{test_set}/test.json")
        img_info = load_img_info(test_set, DATA_DIR["TEST"], img_list)
    else:
        raise NotImplementedError(f"Validation test dataset {dataset} not implemented!")
    print(f"Validation set {dataset} with {len(img_info)} images.")
    return img_info


class DnDataset(ImageBaseDataset):
    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:
        if stage == TRAIN:
            self.patch_size = cfg.patch_size
            self.noise_sigma_range = cfg.noise_sigma_range
            self.img_info = get_train_file(cfg.dataset)  # [:800]
        else:
            self.img_info = get_test_file(cfg.dataset)
        self.noise_sigma = cfg.noise_sigma
        super(DnDataset, self).__init__(cfg, stage, num_train_samples)

    def __getitem__(self, index: int):

        # load image and sample a patch
        index = self._get_index(index)
        img_gt = self._load_item(index)
        img_gt = self._sample_patch(img_gt)
        img_gt = self._augment(img_gt)

        img_gt = np.ascontiguousarray(img_gt)
        img_gt = transforms.functional.to_tensor(img_gt)

        # Add noise
        if self.stage == TRAIN:
            if len(self.noise_sigma_range) > 0:
                noise_sigma = np.random.uniform(*self.noise_sigma_range) / 255
            else:
                noise_sigma = self.noise_sigma / 255
            noise = torch.randn_like(img_gt) * noise_sigma
        else:
            noise_sigma = self.noise_sigma / 255
            img_name = self.img_info[index][0].split("_")[0]
            seed = np.frombuffer(
                sha256(img_name.encode("utf-8")).digest(), dtype="uint32"
            )
            rstate = np.random.RandomState(seed)
            noise = rstate.normal(0, noise_sigma, img_gt.shape)
            noise = torch.from_numpy(noise).float()

        img_lq = img_gt + noise
        if self.cfg.noise_level_map:
            noise_level_map = torch.ones((1, *img_gt.shape[1:]))
            noise_level_map = noise_level_map.mul_(noise_sigma).float()
            img_lq = torch.cat((img_lq, noise_level_map), 0)

        return {
            "indices": index,
            "img_lq": img_lq,
            "img_gt": img_gt,
            "filenames": self.img_info[index][0],
        }

    # def _sample_patch(self, image: np.ndarray) -> np.ndarray:
    #     if self.stage == TRAIN:
    #         # pad the image is necessary
    #         h, w, c = image.shape
    #         if h < self.patch_size or w < self.patch_size:
    #             h_pad = max(0, self.patch_size - h)
    #             w_pad = max(0, self.patch_size - w)
    #             img_pad = ((0, h_pad), (0, w_pad), (0, 0))
    #             image = np.pad(image, img_pad, "constant", constant_values=0)

    #         # sample patch while training
    #         x = random.randrange(0, image.shape[0] - self.patch_size + 1)
    #         y = random.randrange(0, image.shape[1] - self.patch_size + 1)
    #         image = image[x : x + self.patch_size, y : y + self.patch_size]
    #     else:
    #         # TODO: whether need to be multiples of 8?
    #         x = image.shape[0] // self.cfg.modulo * self.cfg.modulo
    #         y = image.shape[1] // self.cfg.modulo * self.cfg.modulo
    #         image = image[:x, :y]
    #     return image


# class DnTrainDataset(DnBaseDataset):
#     def __init__(self, cfg: DictConfig, stage: str) -> None:

#         div2k_files = [
#             (f"{i:04d}.png", osp.join(DIV2K_DIR, f"{i:04d}.png"))
#             for i in range(1, 801)
#         ]
#         assert len(div2k_files) == 800

#         flickr2k_files = [
#             (f"{i:06d}.png", osp.join(FLICKR2K_DIR, f"{i:06d}.png"))
#             for i in range(1, 2651)
#         ]
#         assert len(flickr2k_files) == 2650

#         bsd_files = [(f, osp.join(BSD400_DIR, f)) for f in pathmgr.ls(BSD400_DIR)]
#         assert len(bsd_files) == 400

#         wed_files = [
#             (f"{i:05d}.png", osp.join(WED_DIR, f"{i:05d}.bmp"))
#             for i in range(1, 4745)
#         ]
#         assert len(wed_files) == 4744

#         files = div2k_files + flickr2k_files + bsd_files + wed_files
#         super(DnTrainDataset, self).__init__(cfg, stage, files)


# class CBSD68Dataset(DnBaseDataset):
#     def __init__(self, cfg: DictConfig, stage: str) -> None:
#         files = [(f, osp.join(CBSD68_DIR, f)) for f in pathmgr.ls(CBSD68_DIR)]
#         assert len(files) == 68
#         super(CBSD68Dataset, self).__init__(cfg, stage, files)


# class Kodak24Dataset(DnBaseDataset):
#     def __init__(self, cfg: DictConfig, stage: str) -> None:
#         files = [(f, osp.join(KODAK24_DIR, f)) for f in pathmgr.ls(KODAK24_DIR)]
#         assert len(files) == 24
#         super(Kodak24Dataset, self).__init__(cfg, stage, files)


# class McMasterDataset(DnBaseDataset):
#     def __init__(self, cfg: DictConfig, stage: str) -> None:
#         files = [(f, osp.join(MCMASTER_DIR, f)) for f in pathmgr.ls(MCMASTER_DIR)]
#         assert len(files) == 18
#         super(McMasterDataset, self).__init__(cfg, stage, files)


# class Urban100Dataset(DnBaseDataset):
#     def __init__(self, cfg: DictConfig, stage: str) -> None:
#         files = [(f, osp.join(URBAN100_DIR, f)) for f in pathmgr.ls(URBAN100_DIR)]
#         assert len(files) == 100
#         super(Urban100Dataset, self).__init__(cfg, stage, files)


# class Set12Dataset(DnBaseDataset):
#     def __init__(self, cfg: DictConfig, stage: str) -> None:
#         files = [(f, osp.join(SET12_DIR, f)) for f in pathmgr.ls(SET12_DIR)]
#         assert len(files) == 12
#         super(Set12Dataset, self).__init__(cfg, stage, files)


# class BSD68Dataset(DnBaseDataset):
#     def __init__(self, cfg: DictConfig, stage: str) -> None:
#         files = [(f, osp.join(BSD68_DIR, f)) for f in pathmgr.ls(BSD68_DIR)]
#         assert len(files) == 68
#         super(BSD68Dataset, self).__init__(cfg, stage, files)
