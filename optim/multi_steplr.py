import torch
from torch.optim.lr_scheduler import MultiStepLR


class MultiStepLRWarmup(MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        warmup_iter=-1,
        warmup_init_lr=0,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_init_lr = warmup_init_lr
        super(MultiStepLRWarmup, self).__init__(
            optimizer, milestones, gamma, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch < self.warmup_iter:
            return [
                self.warmup_init_lr
                + (v - self.warmup_init_lr) / self.warmup_iter * self.last_epoch
                for v in self.base_lrs
            ]
        else:
            return super(MultiStepLRWarmup, self).get_lr()


def multi_steplr(optimizer, milestones, gamma, warmup_iter=-1, warmup_init_lr=0):
    if isinstance(milestones, str):
        milestones = list(map(int, milestones.split("+")))
    lr_scheduler = MultiStepLRWarmup(
        optimizer, milestones, warmup_iter, warmup_init_lr, gamma
    )
    return lr_scheduler


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from torch.optim.adamw import AdamW

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = AdamW(model, 2e-4)

    num_steps = 8000
    warmup_iter = 1000
    warmup_init_lr = 1e-5
    gamma = 0.5
    milestones = [3000, 5000, 65000, 7000, 7500]

    lr_scheduler = multi_steplr(optim, milestones, gamma, warmup_iter, warmup_init_lr)

    lrs = []
    for i in range(num_steps):
        optim.step()  # backward pass (update network)
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr()[0])
        print(i, lr_scheduler.get_last_lr()[0], optim.param_groups[0]["lr"])

    plt.plot([i for i in range(num_steps)], lrs)

    plt.legend()
    plt.show()
