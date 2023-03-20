import numpy as np
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
from utils.utils_mosaic import mosaic_CFA_Bayer


class DemosaicDataset(ImageBaseDataset):
    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:
        if stage == TRAIN:
            self.patch_size = cfg.patch_size
            self.img_info = get_train_file(cfg.dataset)  # [:800]
        else:
            self.img_info = get_test_file(cfg.dataset)
        super(DemosaicDataset, self).__init__(cfg, stage, num_train_samples)

    def __getitem__(self, index: int):

        # load image and sample a patch
        index = self._get_index(index)
        img_gt = self._load_item(index)
        img_gt = self._sample_patch(img_gt)
        img_gt = self._augment(img_gt)

        img_lq = mosaic_CFA_Bayer(img_gt)[1]

        img_lq = np.ascontiguousarray(img_lq)
        img_gt = np.ascontiguousarray(img_gt)
        img_lq = transforms.functional.to_tensor(img_lq)
        img_gt = transforms.functional.to_tensor(img_gt)

        return {
            "indices": index,
            "img_lq": img_lq,
            "img_gt": img_gt,
            "filenames": self.img_info[index][0],
        }
        # return index, img_lq, img_gt, self.img_info[index][0]

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
    #         x = image.shape[0] // 8 * 8  # TODO: whether need to be multiples of 8?
    #         y = image.shape[1] // 8 * 8
    #         image = image[:x, :y]
    #     return image
