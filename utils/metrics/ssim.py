import math
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from utils.metrics.psnr import average_metric
from utils.utils_image import rgb2ycbcr, tensor2uint
from torch.autograd import Variable


# ----------
# SSIM based on torch
# ----------
def gaussian(window_size, sigma):
    kernel = [
        round(math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)), 6)
        for x in range(window_size)
    ]
    kernel = np.asarray(kernel)
    gauss = torch.from_numpy(kernel)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(
    img1, img2, window, window_size, channel, size_average: bool = True, mask=None
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if mask is not None:
        ssim_map *= mask
        return ssim_map.sum() / mask.sum()

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size: int = 11, size_average: bool = True):
    if img2.dim() == 5:  # BNCHW
        B, T, C, H, W = img2.size()
        img2 = img2.contiguous().view(B * T, C, H, W)
        img1 = img1.contiguous().view(B * T, C, H, W)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# ----------
# SSIM based on cv2 and numpy
# ----------
# def calculate_ssim(img1, img2, border=0):
#     """calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     """
#     if not img1.shape == img2.shape:
#         raise ValueError("Input images must have the same dimensions.")
#     h, w = img1.shape[:2]
#     img1 = img1[border : h - border, border : w - border]
#     img2 = img2[border : h - border, border : w - border]

#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError("Wrong input image dimensions.")


# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
#         (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#     )
#     return ssim_map.mean()


class StructuralSimilarityIndexMeasure(torchmetrics.Metric):
    def __init__(
        self,
        border: Optional[int] = 0,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        reduction: str = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        compute_on_step: Optional[bool] = True,
        channel="rgb",
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.border = border
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction
        self.channel = channel

        self.add_state("value", default=[], dist_reduce_fx="cat")
        self.add_state("idx", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, idx=List) -> None:

        if self.channel == "y":
            # Pay attention to the in-place in the previous version of rgb2ycbcr
            preds = rgb2ycbcr(preds, self.data_range)
            target = rgb2ycbcr(target, self.data_range)

        assert preds.shape == target.shape
        value = []
        for p, t in zip(preds, target):
            # value.append(
            #     torchmetrics.functional.structural_similarity_index_measure(
            #         p.unsqueeze(0),
            #         t.unsqueeze(0),
            #         self.kernel_size,
            #         self.sigma,
            #         self.reduction,
            #         self.data_range,
            #         self.k1,
            #         self.k2,
            #     )
            # )
            # value.append(ssim(tensor2uint(p, self.data_range), tensor2uint(t, self.data_range)))
            value.append(ssim(p.unsqueeze(0), t.unsqueeze(0)))
        self.value.append(torch.tensor(value).to(preds.device))
        idx = torch.as_tensor(idx, device=preds.device)
        self.idx.append(idx)

    def compute(self) -> torch.Tensor:
        ssim = average_metric(self.value, self.idx)
        # print("ssim values", ssim)
        return ssim


# There seem to be a bug in the ssim function of torchmetrics.
# When the values are calculated on the server, the ssim is about 0.01 lower.
# The following shows the difference.

# Value computed by torchmetrics functions.
# [0]:        val_psnr            25.948076248168945
# [0]:       val_psnr_y           27.486125946044922
# [0]:        val_ssim            0.7963590025901794
# [0]:       val_ssim_y           0.8175588846206665

# Value computed by custom functions.
# [0]:        val_psnr            25.948076248168945
# [0]:       val_psnr_y           27.486127853393555
# [0]:        val_ssim            0.8082676529884338
# [0]:       val_ssim_y           0.8275952935218811
