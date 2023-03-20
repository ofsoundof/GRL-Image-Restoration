from typing import Any, Dict, List

import lpips
import torch
import torchmetrics
from utils.metrics.psnr import average_metric

# from torchmetrics.image import lpip


class LearnedPerceptualImagePatchSimilarity(torchmetrics.Metric):
    def __init__(
        self,
        net_type: str = "vgg",
        reduction: str = "mean",
        compute_on_step: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        self.loss_fn = lpips.LPIPS(net=net_type)
        self.add_state("value", default=[], dist_reduce_fx="cat")
        self.add_state("idx", default=[], dist_reduce_fx="cat")
        # print(self.net)

    def update(self, img1: torch.Tensor, img2: torch.Tensor, idx: List) -> None:
        """Update internal states with lpips score.
        Args:
            img1: tensor with images of shape ``[N, 3, H, W]``
            img2: tensor with images of shape ``[N, 3, H, W]``
        """
        # if not (lpip._valid_img(img1) and lpip._valid_img(img2)):
        #     raise ValueError(
        #         "Expected both input arguments to be normalized tensors (all values in range [-1,1])"
        #         f" and to have shape [N, 3, H, W] but `img1` have shape {img1.shape} with values in"
        #         f" range {[img1.min(), img1.max()]} and `img2` have shape {img2.shape} with value"
        #         f" in range {[img2.min(), img2.max()]}"
        #     )

        loss = self.loss_fn.forward(img1, img2, normalize=True).squeeze()
        # print(loss)
        self.value.append(loss)
        idx = torch.as_tensor(idx, device=img1.device)
        self.idx.append(idx)

        # self.sum_scores += loss.sum()
        # self.total += img1.shape[0]

    def compute(self) -> torch.Tensor:
        lpips = average_metric(self.value, self.idx)
        return lpips
