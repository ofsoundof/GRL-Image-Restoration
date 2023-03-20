"""
From Restormer
"""
import math

from torch.optim.lr_scheduler import _LRScheduler


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(
        self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert len(self.periods) == len(
            self.restart_weights
        ), "periods and restart_weights should have the same length."
        self.cumulative_period = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min
            + current_weight
            * 0.5
            * (base_lr - eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":
    import torch
    from matplotlib import pyplot as plt
    from torch.optim.adamw import AdamW

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = AdamW(model, 1e-4)
    num_steps = 300000

    lr_scheduler = CosineAnnealingRestartCyclicLR(
        optim, [200000, 100000], [1, 0.5], [1e-6, 1e-6]
    )

    lrs = []
    for i in range(num_steps):
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr())
    plt.plot([i for i in range(num_steps)], lrs, label="With warmup_prefix")
    plt.show()
