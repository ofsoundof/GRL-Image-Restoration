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
from utils.utils_deblur import get_blur_kernel


class DeblurDataset(ImageBaseDataset):
    def __init__(self, cfg: DictConfig, stage: str, num_train_samples: int = 0) -> None:
        if stage == TRAIN:
            blur_kernel = get_blur_kernel(cfg.kernel_type)
            self.patch_size = cfg.patch_size + blur_kernel.shape[2] - 1
            self.img_info = get_train_file(cfg.dataset)  # [:800]
        else:
            self.img_info = get_test_file(cfg.dataset)
        self.noise_sigma = cfg.noise_sigma / 255.0

        super(DeblurDataset, self).__init__(cfg, stage, num_train_samples)

    def __getitem__(self, index: int):

        # load image and sample a patch
        index = self._get_index(index)
        img_gt = self._load_item(index)
        img_gt = self._sample_patch(img_gt)
        img_gt = self._augment(img_gt)

        img_gt = np.ascontiguousarray(img_gt)
        img_gt = transforms.functional.to_tensor(img_gt)

        if self.stage != TRAIN:
            np.random.seed(seed=0)  # for reproducibility
        noise = np.random.normal(0, self.noise_sigma, img_gt.shape)  # add AWGN
        noise = torch.from_numpy(noise.astype(np.float32))  # add noise to blurred image

        return {
            "indices": index,
            "img_lq": noise,
            "img_gt": img_gt,
            "filenames": self.img_info[index][0],
        }
        # return index, noise, img_gt, self.img_info[index][0]

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
