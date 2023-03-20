import copy
import logging
import os
import os.path as osp
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from utils.utils_train import (
    find_last_checkpoint_path,
    get_train_samples,
    parse_csv,
)
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

logger: Logger = logging.getLogger(__name__)


@dataclass
class TrainOutput:
    checkpoint_dir: Optional[str] = None
    tensorboard_log_dir: Optional[str] = None


def train(cfg: DictConfig) -> TrainOutput:

    # Get the the information of the allocated devices
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))

    # TODO: pay attention to seed
    # ######### Set Seeds ###########
    seed_everything(cfg.seed)

    ### tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=osp.expanduser(cfg.io.base_output_path)
        if cfg.training
        else os.path.join(osp.expanduser(cfg.io.base_output_path), "TEST"),
        name=cfg.tag,
        version=cfg.io.version,
    )

    ######### DataLoader ###########
    print("===> Loading datasets")
    print(cfg.data_module)
    data_module = []
    # Multiple validation set.
    for val_dataset in cfg.data_module.val.dataset.split("+"):
        data_moduel_cfg = copy.deepcopy(cfg.data_module)
        data_moduel_cfg.val.dataset = val_dataset
        data_module.append(
            hydra.utils.instantiate(
                data_moduel_cfg, num_train_samples=get_train_samples(cfg)
            )
        )

    print("===> Start buildng model")
    ######### Instantiate engine ###########
    model = hydra.utils.instantiate(cfg.engine, cfg)
    # Pay attention to two things:
    #   1. DictConfig could be used as an ordinary dictionary. Thus, the following two syntaxes to access dict elements are OK.
    #       d = DictConfig({'key1': 'value1', 'key2': 'value2'}); d.key1; d['key1']
    #   2. When opening checkpoint files with open method, set mode to "rb".
    if cfg.pretrained_checkpoint is not None:
        if not cfg.load_state_dict:
            print(f"Loading pretrained_checkpoint from {cfg.pretrained_checkpoint}")
            model = model.load_from_checkpoint(
                checkpoint_path=cfg.pretrained_checkpoint
            )
            # cfg_copy = {
            #     "training": cfg.training,
            #     "trainer": cfg.trainer,
            #     "data_module": cfg.data_module,
            # }
            # model = model.load_from_checkpoint(
            #     checkpoint_path=cfg.pretrained_checkpoint, **cfg_copy, strict=False
            # )
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/5337
            # Note that the checkpoint in `checkpoint_path` should contain all of the keys in cfg

        else:
            print(f"Loading the state_dict from {cfg.pretrained_checkpoint}")
            state_dict = torch.load(
                open(cfg.pretrained_checkpoint, "rb"), map_location="cpu"
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                state_dict.pop("current_val_metric")
                state_dict.pop("best_val_metric")
                state_dict.pop("best_iter")
            if hasattr(model.model, "convert_checkpoint"):
                state_dict = model.model.convert_checkpoint(state_dict)
            if "params" in state_dict:
                state_dict = state_dict["params"]
            try:
                current_state_dict = model.state_dict()
                current_state_dict.update(state_dict)
                model.load_state_dict(current_state_dict, strict=True)
            except BaseException:
                # new_state_dict = {}
                # for k, v in state_dict.items():
                #     if k.find("model.") >= 0:
                #         new_state_dict[k.replace("model.", "")] = v
                # model.model.load_state_dict(new_state_dict, strict=True)
                model.model.load_state_dict(state_dict, strict=True)

    ######### Checkpoints ###########
    checkpoint_dirpath = f"{tb_logger.log_dir}/checkpoints"
    print(f"Checkpoint path: {checkpoint_dirpath}")
    last_ckpt = None
    if cfg.resume:
        last_ckpt = find_last_checkpoint_path(checkpoint_dirpath)
    print("===> Checkpoint callbacks")
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dirpath, **cfg.model_checkpoint
    )
    callbacks = [model_checkpoint]

    print("===> Instantiate other callbacks")
    ######### Instantiate callbacks #########
    for _, callback in cfg.callbacks.items():
        callbacks.append(hydra.utils.instantiate(callback))

    ######### Trainer #########
    if cfg.trainer.strategy == "ddp":
        del cfg.trainer.strategy
        trainer = Trainer(
            **cfg.trainer,
            logger=tb_logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=cfg.find_unused_parameters),
        )
    else:
        trainer = Trainer(
            **cfg.trainer,
            logger=tb_logger,
            callbacks=callbacks,
        )

    ######### Start training or validation #########
    val_sets = cfg.data_module.val.dataset

    if cfg.training:
        print("===> Start training")
        # print(model)
        # print(data_module)
        model.hparams.data_module.val.dataset = val_sets.split("+")[0]
        # print("Validation set", val_sets.split("+")[0])
        trainer.fit(model, data_module[0], ckpt_path=last_ckpt)

    print("===> Start validation")
    model.final_validate = True
    for d, v in zip(data_module, val_sets.split("+")):
        model.hparams.data_module.val.dataset = v
        # print("Validation set", v)
        trainer.validate(model, d)
    parse_csv(tb_logger.log_dir)

    if not cfg.training:
        trainer.save_checkpoint(f"{checkpoint_dirpath}/last.ckpt")
    # if cfg.model.name != "burst":
    #     get_flops_params(model, cfg.in_channels, cfg.data_module.train.patch_size)

    return TrainOutput(
        checkpoint_dir=model_checkpoint.dirpath if model_checkpoint else None,
        tensorboard_log_dir=tb_logger.log_dir if tb_logger else None,
    )


@hydra.main(config_path="../config", config_name="defaults", version_base="1.1")
def run(cfg: DictConfig) -> Dict:
    print(f"PyTorch-Lightning Version: {pl.__version__}")

    os.environ["HYDRA_FULL_ERROR"] = os.environ.get("HYDRA_FULL_ERROR", "1")
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print(OmegaConf.to_yaml(cfg, resolve=True))
    return train(cfg).__dict__


if __name__ == "__main__":
    run()
