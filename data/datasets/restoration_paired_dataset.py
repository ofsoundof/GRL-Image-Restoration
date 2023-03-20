from typing import List, Tuple

import numpy as np
import torchvision.transforms as transforms
from omegaconf import DictConfig
from data.datasets.base_image import (
    DATA_DIR,
    ImageBaseDataset,
    load_img_info,
    load_json,
    TRAIN,
    VAL,
)


##################
# motion blur
##################


def _get_real_blur(dataset: str, train: bool):
    # dataset: realblur-j, realblur-r
    identifier = "train" if train else "test"
    filename = f"{identifier}_{dataset[-1]}.json"
    dataset = "RealBlur"
    img_list = load_json(f"{dataset}/{filename}")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    return img_info


def _get_hide(train: bool):
    identifier = "train" if train else "test"
    dataset = "HIDE"
    filename = f"{identifier}.json"
    img_list = load_json(f"{dataset}/{filename}")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    return img_info


def _get_gopro(train: bool):
    # GoPro11, test
    # keys = [
    #     "GOPR0384_11_00", "GOPR0384_11_05", "GOPR0385_11_01", "GOPR0396_11_00", "GOPR0410_11_00", "GOPR0854_11_00", "GOPR0862_11_00", "GOPR0868_11_00", "GOPR0869_11_00", "GOPR0871_11_00", "GOPR0881_11_01",
    # ]
    # sequence_lengths = [100, 100, 100, 100, 134, 100, 77, 100, 100, 100, 100]

    # GoPro8, test
    #     keys = ['GOPR0384_11_00','GOPR0384_11_05','GOPR0385_11_01','GOPR0396_11_00','GOPR0410_11_00','GOPR0854_11_00','GOPR0862_11_00','GOPR0868_11_00']
    #     sequence_lengths = [100,100,100,100,134,100,77,100]

    # GoPro1, test
    #     keys = ['GOPR0384_11_00']
    #     sequence_lengths = [100]

    # Train
    # keys = [
    #     "GOPR0372_07_00", "GOPR0372_07_01", "GOPR0374_11_00", "GOPR0374_11_01", "GOPR0374_11_02", "GOPR0374_11_03", "GOPR0378_13_00", "GOPR0379_11_00", "GOPR0380_11_00", "GOPR0384_11_01", "GOPR0384_11_02",
    #     "GOPR0384_11_03", "GOPR0384_11_04", "GOPR0385_11_00", "GOPR0386_11_00", "GOPR0477_11_00", "GOPR0857_11_00", "GOPR0868_11_01", "GOPR0868_11_02", "GOPR0871_11_01", "GOPR0881_11_00", "GOPR0884_11_00",
    # ]
    # sequence_lengths = [100, 75, 150, 80, 100, 48, 110, 100, 60, 100, 100, 100, 100, 100, 100, 80, 100, 100, 100, 100, 100, 100]

    identifier = "train" if train else "test"
    dataset = "GOPRO"
    filename = f"{identifier}.json"
    img_list = load_json(f"{dataset}/{filename}")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    return img_info


##################
# defocus deblur
##################
def _get_dpdd(dataset, split):
    filename = f"{split}_dual.json" if dataset.find("dual") >= 0 else f"{split}.json"
    dataset = "DPDD"
    img_list = load_json(f"{dataset}/{filename}")
    img_info = load_img_info(dataset, DATA_DIR[dataset], img_list)
    return img_info


def get_train_file(dataset):
    dataset = dataset.lower()
    if dataset == "gopro":
        # motion blur
        img_info = _get_gopro(train=True)
    elif dataset == "dpdd" or dataset == "dpdd_dual":
        # defocus deblur
        img_info = _get_dpdd(dataset, "train")
    elif dataset.find("realblur") >= 0:
        img_info = _get_real_blur(dataset, train=True)
    elif dataset == "hide":
        img_info = _get_hide(train=True)
    else:
        raise NotImplementedError(
            f"Paired training image dataset {dataset} not implemented."
        )
    print(f"Training dataset {dataset} with {len(img_info)} images.")
    return img_info


def get_test_file(dataset):
    dataset = dataset.lower()
    if dataset.find("realblur") >= 0:
        # motion blur
        img_info = _get_real_blur(dataset, train=False)
    elif dataset.find("hide") >= 0:
        # motion blur
        img_info = _get_hide(train=False)
    elif dataset == "gopro":
        # motion blur
        img_info = _get_gopro(train=False)
    elif dataset.find("dpdd") >= 0 and dataset.find("test") >= 0:
        # defocus deblur
        img_info = _get_dpdd(dataset, "test")
    elif dataset.find("dpdd") >= 0 and dataset.find("val") >= 0:
        # defoucus deblur
        img_info = _get_dpdd(dataset, "val")
    else:
        raise NotImplementedError(
            f"Paired test image dataset {dataset} not implemented."
        )

    print(f"Test dataset {dataset} with {len(img_info)} images.")

    return img_info


class PairedDataset(ImageBaseDataset):
    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:
        if stage == TRAIN:
            self.patch_size = cfg.patch_size
            self.img_info = get_train_file(cfg.dataset)
        else:
            self.img_info = get_test_file(cfg.dataset)
        self.dual_pixel = cfg.dual_pixel

        super(PairedDataset, self).__init__(cfg, stage, num_train_samples)
        self.use_hdf5 = True

    def __getitem__(self, index: int):

        # load image and sample a patch
        index = self._get_index(index)
        img_lq, img_gt = self._load_item(index)
        img_gt, img_lq = self._sample_patch(img_gt, img_lq)
        if self.dual_pixel:
            img_lq_l, img_lq_r, img_gt = self._augment(img_lq + [img_gt])

            img_gt = np.ascontiguousarray(img_gt)
            img_lq_l = np.ascontiguousarray(img_lq_l)
            img_lq_r = np.ascontiguousarray(img_lq_r)
            img_gt = transforms.functional.to_tensor(img_gt)
            img_lq_l = transforms.functional.to_tensor(img_lq_l)
            img_lq_r = transforms.functional.to_tensor(img_lq_r)

            return {
                "indices": index,
                "img_lq_l": img_lq_l,
                "img_lq_r": img_lq_r,
                "img_gt": img_gt,
                "filenames": self.img_info[index][0],
            }
        else:
            img_lq, img_gt = self._augment([img_lq, img_gt])

            img_gt = np.ascontiguousarray(img_gt)
            img_lq = np.ascontiguousarray(img_lq)
            img_gt = transforms.functional.to_tensor(img_gt)
            img_lq = transforms.functional.to_tensor(img_lq)

            return {
                "indices": index,
                "img_lq": img_lq,
                "img_gt": img_gt,
                "filenames": self.img_info[index][0],
            }

    def _load_item(self, index: int):
        img_path = self.img_info[index]
        # print(img_path)
        img_gt = self._cache_image(img_path[0:2])
        if self.dual_pixel:
            img_lq_l = self._cache_image(img_path[2:4])
            img_lq_r = self._cache_image(img_path[4:])
            return [img_lq_l, img_lq_r], img_gt
        else:
            img_lq = self._cache_image(img_path[2:])
            return img_lq, img_gt
