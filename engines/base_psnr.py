from engines.base import BaseEngine
from torchvision.transforms.functional import to_pil_image, to_tensor  # noqa


class PSNREngine(BaseEngine):
    """
    Image reconstruction module. Inherits from functionalities from
    LightningModule to decouple boilerplate code from the ML logic.

    Args:
        hparams: a dictionary with the configuration parameters, see config/defaults.yaml.
    """

    def __init__(self, hparams):
        super(PSNREngine, self).__init__(hparams)
        self._init_engine()

    def _init_engine(self):
        self.model_g = self.model
        self.pixel_loss = self.criterion.l1
        self.use_usm_pixel = self.hparams.data_module.train.use_usm_pixel

    def forward(self, batch):
        input_, target = batch["img_lq"], batch["img_gt"]
        indices, filenames = batch["indices"], batch["filenames"]
        out = self.model_g(input_)

        if "img_gt_usm" in batch:
            return out, input_, target, filenames, indices, batch["img_gt_usm"]
        else:
            return out, input_, target, filenames, indices

    def training_step(self, batch, batch_idx):
        restored, input_, target, filenames, indices, target_usm = self(batch)
        loss = self.compute_loss(restored, target, target_usm)
        loss.update({"restored": restored, "target": target})
        return loss

    def compute_loss(self, restored, target, target_usm):
        # pixel loss
        t = target_usm if self.use_usm_pixel else target
        loss = self.pixel_loss.weight * self.pixel_loss.loss_func(restored, t)
        return {"loss": loss}

    def training_step_end(self, step_output):
        self.log_dict(
            self.train_metrics(step_output.pop("restored"), step_output.pop("target")),
            prog_bar=True,
        )
        # step_output.pop("loss")
        # if len(step_output) > 0:
        #     self.log_dict(step_output, prog_bar=True)
