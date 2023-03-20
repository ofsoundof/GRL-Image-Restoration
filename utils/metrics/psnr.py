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

import torch
import torchmetrics
from utils.utils_image import rgb2ycbcr


def average_metric(metric, idx):
    # tensors are not gathered during the steps. Thus, metric and idx remains as lists.
    # after the epochs, the tensors are gathered.
    device = None
    if isinstance(metric, list):
        device = metric[0].device
        metric = torch.cat([m.to(device) for m in metric])
    if isinstance(idx, list):
        if device is None:
            device = idx[0].device
        idx = torch.cat([i.to(device) for i in idx])
    unique_idx = []
    unique_metric = []
    # if strategy == "dp":
    #     metric = torch.cat([m.to("cuda:0") for m in metric])
    #     idx = torch.cat([i.to("cuda:0") for i in idx])
    # print("average_metric", len(metric), len(idx))
    # print(metric)
    for m, i in zip(metric, idx):
        if i not in unique_idx:
            unique_idx.append(i)
            unique_metric.append(m)
    return sum(unique_metric) / len(unique_metric)


def psnr(restored, target):
    diff = restored - target
    mse = diff.pow(2)
    mse = mse.mean([-3, -2, -1])
    return -10 * mse.log10()


class PeakSignalNoiseRatio(torchmetrics.Metric):
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
        self.value.append(psnr(preds, target))
        idx = torch.as_tensor(idx, device=preds.device)
        self.idx.append(idx)

    def compute(self) -> torch.Tensor:
        psnr = average_metric(self.value, self.idx)
        return psnr
