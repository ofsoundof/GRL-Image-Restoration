import bisect
import csv
import os
import os.path as osp
import random
import time

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from utils import dataset_utils
from utils.utils_deblur import get_blur_kernel
from utils.utils_image import (
    img_from_prob_argmax,
    img_from_prob_weighted,
    img_to_class,
    img_to_prob,
    shave,
    tensor_round,
)
from utils.utils_mosaic import dm_matlab
from timm.scheduler.scheduler import Scheduler
from torchmetrics import MetricCollection
from torchvision.transforms.functional import to_pil_image, to_tensor  # noqa


class BaseEngine(pl.LightningModule):
    """
    Image reconstruction module. Inherits from functionalities from
    LightningModule to decouple boilerplate code from the ML logic.

    Args:
        hparams: a dictionary with the configuration parameters, see config/defaults.yaml.
    """

    def __init__(self, hparams):
        super(BaseEngine, self).__init__()
        self.save_hyperparameters(hparams)
        self.final_validate = False
        self.val_results = ""

        self.model = hydra.utils.instantiate(self.hparams.model)
        if self.global_rank == 0 and self.hparams.print_model:
            print(self.model)
        self.criterion = hydra.utils.instantiate(self.hparams.loss)
        metrics = {
            k: hydra.utils.instantiate(v) for k, v in self.hparams.metric.items()
        }
        self.train_metrics = MetricCollection(
            {k: v for k, v in metrics.items() if k.find("train") >= 0}
        )
        self.val_metrics = MetricCollection(
            {k: v for k, v in metrics.items() if k.find("val") >= 0}
        )
        self.register_buffer("current_val_metric", torch.zeros(len(self.val_metrics)))
        self.register_buffer("best_val_metric", torch.zeros(len(self.val_metrics)))
        self.register_buffer(
            "best_iter", torch.zeros(len(self.val_metrics), dtype=torch.int64)
        )

        # Epoch-based or step-based training
        self.epoch_based = self.hparams.trainer.val_check_interval is None

        if self.hparams.data_module.name == "db":
            self.blur_kernel = get_blur_kernel(self.hparams.data_module.kernel_type)

        if self.hparams.data_module.name.find("sr") >= 0:
            self.scale = self.hparams.data_module.scale
        else:
            self.scale = 1
        steps = self.hparams.steps
        self.hparams.steps = [sum(steps[0 : i + 1]) for i in range(0, len(steps))]

        # with open(f"{self.logger.log_dir}/log.txt", "a") as f:
        #     f.write(str(self.hparams) + "\n")
        #     f.write(str(self.model) + "\n")

        # macs, params = get_model_complexity_info(
        #     self.model,
        #     (3, self.hparams.train_ps, self.hparams.train_ps),
        #     as_strings=True,
        #     print_per_layer_stat=False,
        #     verbose=True,
        # )
        # print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        # print("{:<30}  {:<8}".format("Number of parameters: ", params))

    def forward_tile(self, input_):
        # Test the image tile by tile
        b, _, h, w = input_.size()
        c = self.hparams.data_module.num_channels
        tile = min(self.hparams.tile, h, w)
        tile_overlap = self.hparams.tile_overlap

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * self.scale, w * self.scale).type_as(input_)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                out_patch = self.model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                x0 = h_idx * self.scale
                y0 = w_idx * self.scale
                x1 = (h_idx + tile) * self.scale
                y1 = (w_idx + tile) * self.scale

                E[..., x0:x1, y0:y1].add_(out_patch)
                W[..., x0:x1, y0:y1].add_(out_patch_mask)
        output = E.div_(W)
        return output

    def forward(self, batch):
        if self.hparams.data_module.get("dual_pixel", False):
            input_ = torch.cat([batch["img_lq_l"], batch["img_lq_r"]], dim=1)
        else:
            input_ = batch["img_lq"]
        target = batch["img_gt"]
        indices, filenames = batch["indices"], batch["filenames"]

        # preprocessing for image demosaicking
        if self.hparams.data_module.name == "dm":
            input_ = dm_matlab(input_)

        # preprocessing for image deblurring
        if self.hparams.data_module.name == "db":
            # for image deblurring, input_ is the noise image, the blurred image is generated in the following.
            bkh, bkw = [s // 2 for s in self.blur_kernel.shape[2:]]
            input_ += F.conv2d(
                target,
                self.blur_kernel.to(input_.device, input_.dtype),
                groups=3,
                padding=(bkh, bkw),
            )
            if self.training:
                input_ = input_[:, :, bkh:-bkh, bkw:-bkw]
                target = target[:, :, bkh:-bkh, bkw:-bkw]

        # sample different batch sizes and pathc sizes. Used by Restormer
        if self.training and len(self.hparams.batch_sizes) > 0:
            train_group = bisect.bisect_left(self.hparams.steps, self.global_step)
            mini_batch_size = self.hparams.batch_sizes[train_group]
            if mini_batch_size < self.hparams.batch_size:
                idx = random.sample(
                    range(0, self.hparams.batch_size), k=mini_batch_size
                )
                indices = indices[idx]
                input_ = input_[idx]
                target = target[idx]
                filenames = filenames[idx]

            mini_patch_size = self.hparams.patch_sizes[train_group]
            if mini_patch_size < self.hparams.patch_size:
                x0 = random.randrange(0, self.hparams.patch_size - mini_patch_size + 1)
                y0 = random.randrange(0, self.hparams.patch_size - mini_patch_size + 1)
                x1 = x0 + mini_patch_size
                y1 = y0 + mini_patch_size
                scale = self.scale
                input_ = input_[:, :, x0:x1, y0:y1]
                target = target[:, :, x0 * scale : x1 * scale, y0 * scale : y1 * scale]

        # print(input_.shape, target.shape)
        if self.hparams.mixup and self.training and self.current_epoch > 5:
            input_, target = dataset_utils.MixUp_AUG().aug(input_, target)

        # tiled inference for networks that consumes lots of memory.
        # with torch.cuda.amp.autocast():
        if self.training:
            out = self.model(input_)
        else:
            if self.hparams.tile == 0:
                out = self.model(input_)
            else:
                out = self.forward_tile(input_)
        # out = torch.clamp(out, 0, 1)
        # TODO: whether need to clamp the restored image here.
        return out, input_, target, filenames, indices

    def compute_loss(self, restored, target):
        loss_value = 0
        for loss_name, loss in self.criterion.items():
            if self.hparams.get("classification", False):
                if loss_name in ["l1", "l2"]:
                    restored_img = img_from_prob_weighted(restored)
                    loss_value += loss.weight * loss.loss_func(restored_img, target)
                elif loss_name == "cross_entropy":
                    if self.hparams.one_hot_label:
                        target_label = img_to_class(target)
                    else:
                        target_label = img_to_prob(target)
                    loss_value += loss.weight * loss.loss_func(restored, target_label)
                else:
                    raise NotImplementedError(
                        f"Loss function {loss_name} not implemented."
                    )
            else:
                # print(restored.shape, target.shape)
                loss_value += loss.weight * loss.loss_func(restored, target)
        return loss_value

    def form_images(self, restored, training=True):
        if self.hparams.get("classification", False):
            if self.hparams.prob_to_image == "argmax":
                restored = img_from_prob_argmax(restored)
            elif self.hparams.prob_to_image == "weighted_sum":
                restored = img_from_prob_weighted(restored)
            else:
                raise NotImplementedError(
                    f"Method {self.hparams.prob_to_image} not implemented."
                )
        else:
            if not training:
                restored = tensor_round(restored, 1.0)
        return restored

    def training_step(self, batch, batch_idx):
        # print(self.global_step)
        restored, input_, target, filenames, indices = self(batch)
        loss = self.compute_loss(restored, target)
        restored = self.form_images(restored)
        # Here need to pay attention to the different between self.optimizers(), self.optimizers(False), self.lr_schedulers().optimizer
        # print("Training Learning rate", len(self.optimizers().param_groups), self.optimizers().param_groups[0]['lr'], self.lr_schedulers().get_last_lr()[0], self.lr_schedulers().last_epoch)
        # print(id(self.optimizers()), id(self.optimizers().param_groups[0]), self.optimizers().param_groups[0]['lr'], self.lr_schedulers().get_last_lr()[0], self.global_rank, self.lr_schedulers().last_epoch, "train first")# , self.optimizers())
        # print(id(self.optimizers(False)), id(self.optimizers(False).param_groups[0]), self.optimizers(False).param_groups[0]['lr'], self.lr_schedulers().get_last_lr()[0], self.global_rank, self.lr_schedulers().last_epoch, "train middle following")# , self.optimizers())
        # print(id(self.lr_schedulers().optimizer), id(self.lr_schedulers().optimizer.param_groups[0]), self.lr_schedulers().optimizer.param_groups[0]['lr'], self.global_rank, "train second")

        # # Call lr_scheduler every step
        # if not self.epoch_based:
        #     sch = self.lr_schedulers()
        #     sch.step()
        return {"loss": loss, "restored": restored, "target": target}

    def training_step_end(self, step_output):
        self.log_dict(
            self.train_metrics(step_output.pop("restored"), step_output.pop("target")),
            prog_bar=True,
        )
        # print(f"Allocated: {torch.cuda.memory_allocated(0)/1024 ** 3: 2.2f}")
        # print(f"Reserved: {torch.cuda.memory_reserved(0)/1024 ** 3: 2.2f}")
        # print(f"Max Reserved: {torch.cuda.max_memory_reserved(0)/1024 ** 3: 2.2f}")

    # def training_epoch_end(self, outputs):
    #     This function causes memory leak.
    #     self.train_metrics.reset()
    #     del outputs
    #     torch.cuda.empty_cache()
    #     # pass
    # def training_epoch_end(self, outputs):
    #     pass

    def validation_step(self, batch, batch_idx):
        # print(batch["img_lq"].shape)
        restored, input_, target, filenames, indices = self(batch)
        restored = self.form_images(restored, False)
        input_ = tensor_round(input_, 1.0)
        target = tensor_round(target, 1.0)

        if self.hparams.save_images:
            self._save_images(input_, restored, target, filenames)
        if self.hparams.data_module.name.find("sr") >= 0:
            restored = shave(restored, self.hparams.data_module.scale)
            target = shave(target, self.hparams.data_module.scale)
        return restored, target, indices, filenames

    def validation_step_end(self, step_output):
        self.val_metrics.update(*step_output[:3])
        return step_output[3]

    def validation_epoch_end(self, outputs):
        # print("Validation", self.global_step)
        # Pytorch-Lighting support automatic checkpoint resume during training.
        # However, the global step is increased by 1 after every break and resume of the training.

        # for k, m in self.val_metrics.items():
        #     print(k)
        #     print(m)
        #     print(len(m.value), len(m.idx))
        # print("Validation Learning rate", self.optimizers().param_groups[0]['lr'], self.lr_schedulers().get_last_lr()[0], self.lr_schedulers().last_epoch)

        metric_one = list(self.val_metrics.values())[0]
        if len(metric_one.value) > 0 and len(metric_one.idx) > 0:
            # This is needed to resume the training.
            metric = self.val_metrics.compute()
            self.update_validation_info(metric, outputs)
            self.log_dict(metric, prog_bar=True)
            self.val_metrics.reset()
            # except BaseException:
        else:
            pass
        # print("epoch info")
        # metric = self.val_metrics.compute()
        # self.log_dict(metric, prog_bar=True)
        # self.val_metrics.reset()
        # print("psnr_metric", psnr_metric)
        # print("Test reset mode: before")
        # print(self.metric.compute(self.hparams.trainer.strategy))
        # print(self.metric.psnr)
        # print(self.metric.idx)
        # self.metric.reset()
        # print("Test reset mode: after")
        # # print(self.metric.compute(self.hparams.trainer.strategy))
        # print(self.metric.psnr)
        # print(self.metric.idx)
        # # print(outputs)
        # # Gather results from all devices.
        # if self.hparams.trainer.strategy == "ddp":
        #     # print("Before")
        #     # pprint(outputs)
        #     outputs = self.all_gather(outputs)
        #     # print("After")
        #     # pprint(outputs)
        #     outputs = [torch.sum(t) for t in outputs]
        # # Calculate every results
        # self.current_psnr = sum(outputs) / len(self.trainer.datamodule.val_dataset)

    def print_per_image_metric(self, filenames):
        # print(filenames)
        # for v in self.val_metrics.values():
        #     print(v.value)
        s = "Filename\t" + "\t".join(self.val_metrics.keys()) + "\n"
        for i in range(len(filenames)):
            s += f"{filenames[i][0]:20}\t"
            for v in self.val_metrics.values():
                # print(v.value[i].detach().cpu().item())
                s += f"{v.value[i].item():.4f}\t"
            s += "\n"
        # print(s)
        val_set = self.hparams.data_module.val.dataset
        path = f"{self.logger.log_dir}/metric_{val_set}_{self.global_rank}.txt"
        with open(path, "a") as f:
            f.write(s)

    def _iter_string(self, log_iter):
        if isinstance(log_iter, torch.Tensor):
            log_iter = log_iter.item()
        if self.epoch_based:
            s = f"Epoch {log_iter:4d} "
        else:
            s = f"Step {round(log_iter / 1000):4d}k "
        return s

    def update_validation_info(self, metric, filenames):
        if self.final_validate:
            self.print_per_image_metric(filenames)

        if self.global_rank == 0:
            # write validation log
            mem = torch.cuda.max_memory_allocated() / 1024.0**3
            s = f"[{self._iter_string(self.current_epoch if self.epoch_based else self.global_step)}"
            if self.hparams.training:
                if isinstance(self.lr_schedulers(), list):
                    lr = self.lr_schedulers()[0].optimizer.param_groups[0]["lr"]
                else:
                    lr = self.lr_schedulers().optimizer.param_groups[0]["lr"]
                s += f"LR {lr:.8f} "
            if hasattr(self, "model_g"):
                num_param = [
                    p.numel() for p in self.model_g.parameters() if p.requires_grad
                ]
            else:
                num_param = [
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ]
            num_param = sum(num_param) / 10**6
            s += f"Memory {mem:2.2f}GB Params {num_param:3.2f}M] ---- Train {self.hparams.data_module.train.dataset} / Test {self.hparams.data_module.val.dataset}"
            for i, (k, v) in enumerate(metric.items()):
                self.current_val_metric[i] = v
                s += f"---- {k} {self.current_val_metric[i]:.4f} "
                if self.hparams.training:
                    if self.current_val_metric[i] >= self.best_val_metric[i]:
                        self.best_val_metric[i] = self.current_val_metric[i]
                        self.best_iter[i] = torch.tensor(
                            self.current_epoch
                            if self.epoch_based
                            else self.global_step,
                            dtype=self.best_iter.dtype,
                            device=self.best_iter.device,
                        )
                    s += f"Best {self.best_val_metric[i]:.4f} @ {self._iter_string(self.best_iter[i])}"
            s += "\n"
            print(s)
            log_file_name = "log_final_validate" if self.final_validate else "log"
            print(f"{self.logger.log_dir}/{log_file_name}.txt")
            with open(f"{self.logger.log_dir}/{log_file_name}.txt", "a") as f:
                f.write(s)

            if self.final_validate:
                # write to csv file
                csv_file = f"{self.logger.log_dir}/log_final_validate.csv"
                if not osp.exists(csv_file):
                    with open(csv_file, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Test Set"] + list(metric.keys()))

                with open(csv_file, "a") as f:
                    writer = csv.writer(f)
                    val_set = self.hparams.data_module.val.dataset
                    writer.writerow([val_set] + [v.item() for v in metric.values()])

                # write metrics of every image
                s = ""
                filenames = []
                for i in range(self.trainer.world_size):
                    path = f"{self.logger.log_dir}/metric_{val_set}_{i}.txt"
                    with open(path, "r") as f:
                        lines = f.readlines()
                    if i == 0:
                        s += lines[0]
                    for line in lines[1:]:
                        filename = line.split("\t")[0]
                        if filename not in filenames:
                            filenames.append(filename)
                            s += line.replace(".png", "")
                    os.system(f"rm {path}")
                    path = f"{self.logger.log_dir}/metric_{val_set}.txt"
                    with open(path, "w") as f:
                        f.write(s)

                # file_path = f"{self.logger.log_dir}/validation_results.txt"
                # if not osp.exists(file_path):
                #     with pathmgr.open(file_path, "a") as f:
                #         column_name = "Test Set\t"
                #         for k in metric.keys():
                #             column_name += k + "\t"
                #         f.write(column_name + "\n")

                # with open(file_path, "a") as f:
                #     s = self.hparams.data_module.val.dataset + "\t"
                #     for v in metric.values():
                #         s += f"{v:.4f}\t"
                #     f.write(s + "\n")

    ### hooks

    def configure_optimizers(self):
        """
        Define optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            [optimizer],[scheduler] - The first list contains optimizers, the
            second contains of LR schedulers (or lr_dict).
        """
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, self.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        print("Print optimizers and schedulers!!")
        print(optimizer)
        print(scheduler)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if isinstance(scheduler, Scheduler):
            # Used for self defined schedulers that require an epoch value
            scheduler.step(self.global_step)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    ### extra functionalities and helper functions
    def log_images(self, tn_input, tn_target, tn_prediction, prefix, step):
        """
        Log the upsampled low-resolution image, the reconstructed image and the
        ground-truth to the tensoboard.

        Inputs:
            tn_input (torch.Tensor): a 4D-tensor containing the low-resolution input.
            tn_target (torch.Tensor): a 4D-tensor containing the reconstructed result.
        """
        LOG_N_IMAGES_PER_BATCH = 2

        # interpolate lr image
        B, C, H, W = tn_target.shape
        for i in range(min(B, LOG_N_IMAGES_PER_BATCH)):

            if self.logger.experiment:

                image_grid = torchvision.utils.make_grid(
                    torch.stack([tn_input[i], tn_prediction[i], tn_target[i]])
                )
                self.logger.experiment.add_image(
                    f"{prefix}/{i}", image_grid.clip(0, 1), step
                )

    def _save_images(self, tn_input, tn_prediction, tn_target, filenames):

        # prepare output path
        dataset = filenames[0].split("/")[0]  # batch size is 1 during validation.
        filename = osp.splitext(osp.basename(filenames[0]))[0]

        path = osp.join(self.logger.log_dir, "validation_results")
        if self.hparams.data_module.name.find("sr") >= 0:
            scale = self.hparams.data_module.scale
            path = osp.join(path, f"X{scale}", dataset)
        elif self.hparams.data_module.name.find("dn") >= 0:
            noise_sigma = self.hparams.data_module.noise_sigma
            path = osp.join(path, f"Sigma{noise_sigma}", dataset)
        elif self.hparams.data_module.name.find("jpeg") >= 0:
            quality_factor = self.hparams.data_module.quality_factor
            path = osp.join(path, f"QF{quality_factor}", dataset)
        elif self.hparams.data_module.name.find("paired") >= 0:
            if dataset == "GOPRO":
                subfolder = filenames[0].split("/")[2]
                path = osp.join(path, dataset, subfolder)
            elif dataset == "RealBlur":
                subfolder = osp.join(*filenames[0].split("/")[1:3])
                filename = filename.split("_")[1]
                path = osp.join(path, subfolder)
            else:
                path = osp.join(path, dataset)
        else:
            path = osp.join(path, dataset)

        if not osp.isdir(path):
            os.makedirs(path)

        if self.hparams.data_module.name.find("sr") >= 0:
            tn_input = F.interpolate(tn_input, scale_factor=scale)
        if (
            self.hparams.data_module.name == "dn"
            and self.hparams.data_module.noise_level_map
        ):
            tn_input = tn_input[:, :-1, ...]
        if self.hparams.data_module.get("dual_pixel", False):
            tn_input = tn_input[:, :3, ...]

        # image_grid = torchvision.utils.make_grid(
        #     torch.stack([tn_input[0], tn_prediction[0], tn_target[0]])
        # )
        # to_pil_image(image_grid.detach()).save(
        #     pathmgr.open(clean_path(f"{path}/{filename}.png"), "wb")
        # )
        to_pil_image(tn_input[0].detach()).save(
            open(f"{path}/{filename}_LQ.png", "wb")
        )
        to_pil_image(tn_prediction[0].detach()).save(
            open(f"{path}/{filename}_HQ.png", "wb")
        )
        if self.hparams.save_gt:
            to_pil_image(tn_target[0].detach()).save(
                open(f"{path}/{filename}_GT.png", "wb")
            )

    def on_train_epoch_start(self) -> None:
        """
        Called in the training loop at the very beginning of the epoch.
        Initialize a timer to measure duration of the validation epoch.
        """
        # do something when the epoch starts
        # pyre-fixme[16]: `ImageReconstructionEngine` has no attribute
        #  `epoch_time_meter`.
        self.epoch_time_meter = time.perf_counter()

    def on_train_epoch_end(self) -> None:
        """
        Called in the training loop at the very end of the epoch.
        Log training epoch statistics, e.g. elapsed time.
        """
        # pyre-fixme[58]: `-` is not supported for operand types `float` and
        #  `Union[torch.Tensor, torch.nn.Module]`.
        elapsed_time = time.perf_counter() - self.epoch_time_meter
        # remove in the future
        if self.logger.experiment:
            self.logger.experiment.add_scalar(
                "train_meters/epoch_time", elapsed_time, self.current_epoch
            )

    def on_validation_epoch_start(self):
        """
        Called in the validation loop at the very beginning of the epoch.
        Initialize a timer to measure duration of the validation epoch.
        """
        self.val_time_meter = time.perf_counter()

    def on_validation_epoch_end(self):
        """
        Called in the validation loop at the very end of the epoch.
        Log validation epoch statistics, e.g. elapsed time.
        """
        elapsed_time = time.perf_counter() - self.val_time_meter

        if self.logger.experiment:
            self.logger.experiment.add_scalar(
                "val_meters/epoch_time", elapsed_time, self.current_epoch
            )
