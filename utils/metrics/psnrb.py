"""
How to deal with the case in case of uneven inputs. This the suggestions from the following official website.
Yet, the drawback is that during validation, only one GPU is used. To solve that problem, I did the following
modification of the metrics.

https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-epoch-level-metrics
It is recommended to validate on single device to ensure each sample/batch gets evaluated exactly once.
This is helpful to make sure benchmarking for research papers is done the right way.
Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp".
It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torchmetrics
from utils.metrics.psnr import average_metric
from utils.utils_image import rgb2ycbcr


def blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
        (
            (
                im[:, :, :, block_horizontal_positions]
                - im[:, :, :, block_horizontal_positions + 1]
            )
            ** 2
        )
        .sum(3)
        .sum(2)
        .sum(1)
    )
    vertical_block_difference = (
        (
            (
                im[:, :, block_vertical_positions, :]
                - im[:, :, block_vertical_positions + 1, :]
            )
            ** 2
        )
        .sum(3)
        .sum(2)
        .sum(1)
    )

    nonblock_horizontal_positions = np.setdiff1d(
        torch.arange(0, im.shape[3] - 1), block_horizontal_positions
    )
    nonblock_vertical_positions = np.setdiff1d(
        torch.arange(0, im.shape[2] - 1), block_vertical_positions
    )

    horizontal_nonblock_difference = (
        (
            (
                im[:, :, :, nonblock_horizontal_positions]
                - im[:, :, :, nonblock_horizontal_positions + 1]
            )
            ** 2
        )
        .sum(3)
        .sum(2)
        .sum(1)
    )
    vertical_nonblock_difference = (
        (
            (
                im[:, :, nonblock_vertical_positions, :]
                - im[:, :, nonblock_vertical_positions + 1, :]
            )
            ** 2
        )
        .sum(3)
        .sum(2)
        .sum(1)
    )

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
        n_boundary_horiz + n_boundary_vert
    )

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (
        horizontal_nonblock_difference + vertical_nonblock_difference
    ) / (n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def psnrb(target, input):
    total = 0
    for c in range(input.shape[1]):
        mse = torch.nn.functional.mse_loss(
            input[:, c : c + 1, :, :], target[:, c : c + 1, :, :], reduction="none"
        )
        bef = blocking_effect_factor(input[:, c : c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return total / input.shape[1]


class PeakSignalNoiseRatioBlock(torchmetrics.Metric):
    def __init__(
        self,
        border: Optional[int] = 0,
        data_range: Optional[float] = 1.0,
        base: float = 10.0,
        reduction: str = "none",
        dim: Optional[Union[int, Tuple[int, ...]]] = (1, 2, 3),
        compute_on_step: Optional[bool] = True,
        channel="rgb",
        **kwargs: Dict[str, Any],
    ):
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.border = border
        self.data_range = data_range
        self.base = base
        self.reduction = reduction
        self.dim = dim
        self.channel = channel

        self.add_state("value", default=[], dist_reduce_fx="cat")
        self.add_state("idx", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, idx=List) -> None:

        if self.channel == "y":
            # Pay attention to the in-place in the previous version of rgb2ycbcr
            preds = rgb2ycbcr(preds, self.data_range)
            target = rgb2ycbcr(target, self.data_range)

        assert preds.shape == target.shape
        # print("maximum psnr", preds.max(), target.max())

        # self.value.append(
        #     torchmetrics.functional.peak_signal_noise_ratio(
        #         preds,
        #         target,
        #         data_range=self.data_range,
        #         base=self.base,
        #         reduction=self.reduction,
        #         dim=self.dim,
        #     )
        # )
        self.value.append(psnrb(target, preds))
        idx = torch.as_tensor(idx, device=preds.device)
        self.idx.append(idx)

    def compute(self) -> torch.Tensor:
        psnr = average_metric(self.value, self.idx)
        return psnr
