import csv
import os
from collections import OrderedDict

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from engines.base_psnr import PSNREngine
from utils.utils_image import shave, tensor_round
from torchvision.transforms.functional import to_pil_image, to_tensor  # noqa


class GANEngine(PSNREngine):
    """
    Image reconstruction module. Inherits from functionalities from
    LightningModule to decouple boilerplate code from the ML logic.

    Args:
        hparams: a dictionary with the configuration parameters, see config/defaults.yaml.
    """

    def __init__(self, hparams):
        super(GANEngine, self).__init__(hparams)

    def _init_engine(self):
        self.model_g = self.model.model_g
        self.model_d = self.model.model_d
        if self.hparams.get("bsr_psnr_checkpoint", None) is not None:
            self.load_state_dict_g()
        if self.hparams.get("bsr_discriminator_checkpoint", None) is not None:
            self.load_state_dict_d()

        self.pixel_loss = self.criterion.pixel_loss.loss_func
        self.pixel_loss_weight = self.criterion.pixel_loss.weight

        self.perceptual_loss = self.criterion.perceptual_loss.loss_func
        self.perceptual_loss_weight = self.criterion.perceptual_loss.weight

        self.gan_loss = self.criterion.gan_loss.loss_func
        self.gan_loss_weight = self.criterion.gan_loss.weight

        self.use_usm_pixel = self.hparams.data_module.train.use_usm_pixel
        self.use_usm_percep = self.hparams.data_module.train.use_usm_percep
        self.use_usm_gan = self.hparams.data_module.train.use_usm_gan

    def load_state_dict_g(self):
        # Loading pretrained PSNR-oriented model.
        print("===> Loading pretrained PSNR-oriented model")
        state_dict = torch.load(open(self.hparams.bsr_psnr_checkpoint, "rb"), map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.find("model_g.") >= 0:
                if not (
                    k.find("relative_coords_table") >= 0
                    or k.find("relative_position_index") >= 0
                    or k.find("attn_mask") >= 0
                    or k.find(".table_") >= 0
                    or k.find(".index_") >= 0
                    or k.find(".mask_") >= 0
                    # or k.find(".upsample.") >= 0
                ):
                    new_state_dict[k.replace("model_g.", "")] = v
        current_state_dict = self.model_g.state_dict()
        current_state_dict.update(new_state_dict)
        self.model_g.load_state_dict(current_state_dict, strict=True)

    def load_state_dict_d(self):
        # Loading pretrained PSNR-oriented model.
        print("===> Loading pretrained discriminator model")
        state_dict = torch.load(
            open(self.hparams.bsr_discriminator_checkpoint, "rb"),
            map_location="cpu",
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.find("model_d.") >= 0:
                new_state_dict[k.replace("model_d.", "")] = v
        self.model_d.load_state_dict(new_state_dict, strict=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        restored, input_, target, filenames, indices, target_usm = self(batch)
        loss = self.compute_loss(restored, target, target_usm, optimizer_idx)
        loss.update({"restored": restored, "target": target})
        return loss

    def compute_loss(self, restored, target, target_usm, optimizer_idx):
        # train generator
        loss_g_total = 0
        loss_dict = OrderedDict()
        if optimizer_idx == 0:
            # pixel loss
            if self.pixel_loss:
                t = target_usm if self.use_usm_pixel else target
                loss_g_pix = self.pixel_loss(restored, t)
                loss_g_pix *= self.pixel_loss_weight
                loss_g_total += loss_g_pix
                loss_dict["loss_g_pix"] = loss_g_pix

            # perceptual loss
            if self.perceptual_loss:
                t = target_usm if self.use_usm_percep else target
                loss_g_percep, loss_g_style = self.perceptual_loss(restored, t)
                if loss_g_percep is not None:
                    loss_g_percep *= self.perceptual_loss_weight
                    loss_g_total += loss_g_percep
                    loss_dict["loss_g_percep"] = loss_g_percep
                if loss_g_style is not None:
                    loss_g_style *= self.perceptual_loss_weight
                    loss_g_total += loss_g_style
                    loss_dict["loss_g_style"] = loss_g_style

            # gan loss
            if self.gan_loss:
                fake_g_pred = self.model_d(restored)
                loss_g_gan = self.gan_loss(fake_g_pred, True, is_disc=False)
                loss_g_gan *= self.gan_loss_weight
                loss_g_total += loss_g_gan
                loss_dict["loss_g_gan"] = loss_g_gan

            loss_dict["loss"] = loss_g_total

        # train discriminator
        if optimizer_idx == 1:
            # real
            real_d_pred = self.model_d(target_usm if self.use_usm_gan else target)
            loss_d_real = self.gan_loss(real_d_pred, True, is_disc=True)
            loss_dict["loss_d_real"] = loss_d_real
            loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())

            # fake
            fake_d_pred = self.model_d(restored.detach())
            loss_d_fake = self.gan_loss(fake_d_pred, False, is_disc=True)
            loss_dict["loss_d_fake"] = loss_d_fake
            loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())

            # https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944/2
            loss_d_total = loss_d_real + loss_d_fake

            loss_dict["loss"] = loss_d_total

        return loss_dict

    def validation_step(self, batch, batch_idx):
        restored, input_, target, filenames, indices = self(batch)
        # print(target)
        restored = self.form_images(restored, False)
        input_ = tensor_round(input_, 1.0)
        if target.ndim == 1:
            if self.hparams.save_images:
                self._save_images(
                    input_, restored, torch.zeros_like(restored), filenames
                )
            # if self.hparams.data_module.name.find("sr") > 0:
            #     restored = shave(restored, self.hparams.data_module.scale)
        else:
            target = tensor_round(target, 1.0)
            if self.hparams.save_images:
                self._save_images(input_, restored, target, filenames)
            # if self.hparams.data_module.name.find("sr") > 0:
            #     restored = shave(restored, self.hparams.data_module.scale)
            #     target = shave(target, self.hparams.data_module.scale)
        return restored, target, indices, filenames

    # def validation_epoch_end(self, outputs):
    # disabled: no validation. val_niqe in restorer_niqe.yaml removed, model_checkpoint.monitor=null
    # enabled: with validation. val_niqe in restorer_niqe.yaml exists, model_checkpoint.monitor=niqe
    #     pass

    ### hooks

    def configure_optimizers(self):
        """
        Define optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            [optimizer],[scheduler] - The first list contains optimizers, the
            second contains of LR schedulers (or lr_dict).
        """
        params = self.hparams.optimizer
        optimizer_g = hydra.utils.instantiate(params, self.model_g.parameters())
        optimizer_d = hydra.utils.instantiate(params, self.model_d.parameters())
        optimizers = [optimizer_g, optimizer_d]
        if self.hparams.loss.perceptual_loss.get("requires_grad", False):
            optimizer_l = optim.Adam(
                self.perceptual_loss.parameters(),
                lr=0,
                weight_decay=self.hparams.optimizer.weight_decay,
            )
            optimizers.append(optimizer_l)

        schedulers = [
            {
                "scheduler": hydra.utils.instantiate(self.hparams.lr_scheduler, o),
                "interval": "step",
                "frequency": 1,
            }
            for o in optimizers
        ]

        print("Print optimizers and schedulers!!")
        print(optimizers)
        print(schedulers)
        return optimizers, schedulers
