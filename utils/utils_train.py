import csv
import io
import numpy as np
from omegaconf import DictConfig
from torchvision.transforms.functional import to_pil_image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import Optional
import os
import os.path as osp


def savefig(path, fig):
    import matplotlib.pyplot as plt

    io_buf = io.BytesIO()
    plt.savefig(io_buf, format="raw")
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    # print(img_arr.shape)
    to_pil_image(img_arr).save(path)


def get_train_samples(cfg: DictConfig) -> int:
    if cfg.trainer.val_check_interval is None:
        num_train_samples = 0
    else:
        if cfg.trainer.strategy == "ddp":
            batch_size = cfg.num_nodes * cfg.gpus * cfg.batch_size
        else:
            batch_size = cfg.batch_size

        num_train_samples = batch_size * cfg.trainer.max_steps
    return num_train_samples


def parse_csv(file_path, pruning=False):
    # read csv file
    with open(file_path + "/log_final_validate.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        data = [row for row in csv_reader]
    num_row, num_column = len(data), len(data[0])

    # change the accuracy of PSNR and SSIM
    for i in range(1, num_row):
        for j in range(1, num_column):
            if data[0][j].find("psnr") >= 0:
                data[i][j] = f"{float(data[i][j]):.2f}"
            else:
                data[i][j] = f"{float(data[i][j]):.4f}"

    # print the results table
    s = ""
    for j in range(num_column):
        for i in range(num_row):
            s += data[i][j] + "\t"
        s += "\n"

    # tile PSNR and SSIM
    s += "\n\nChannel\t"
    s += "".join(
        [data[i][0] + " PSNR\t" + data[i][0] + " SSIM\t" for i in range(1, num_row)]
    )
    s += "\n"

    def _tile_psnr_ssim(key1, key2, channel):
        if key1 in data[0] and key2 in data[0]:
            idx1 = data[0].index(key1)
            idx2 = data[0].index(key2)
            s0 = channel + "\t"
            for i in range(1, num_row):
                s0 += data[i][idx1] + "\t" + data[i][idx2] + "\t"
            s0 += "\n"
        else:
            s0 = ""
        return s0

    s += _tile_psnr_ssim("val_psnr", "val_ssim", "RGB")
    s += _tile_psnr_ssim("val_psnr_y", "val_ssim_y", "Y")
    # print(s)

    # Do not use "append" mode when multiple subprocesses tries to write to one file.
    with open(f"{file_path}/validation_results.txt", "w") as f:
        f.write(s)

    if pruning:
        pruning_path = f"{file_path.replace('Finetune', 'Prune')}/compression_ratio.txt"
        with open(pruning_path, "r") as f:
            x = f.readlines()
        s += "\n" + "".join(x)
    with open(f"{file_path}/validation_results.txt", "w") as f:
        f.write(s)

    # import pandas as pd
    # csv = pd.read_csv(file_path)
    # df_csv = pd.DataFrame(data=csv)
    # transposed_csv = df_csv.T
    # print(df_csv)
    # print(transposed_csv)


def find_last_checkpoint_path(checkpoint_dir: Optional[str]) -> Optional[str]:
    if checkpoint_dir is None:
        return None
    checkpoint_file_name = (
        f"{ModelCheckpoint.CHECKPOINT_NAME_LAST}{ModelCheckpoint.FILE_EXTENSION}"
    )
    last_checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_file_name)
    if not osp.exists(last_checkpoint_filepath):
        return None

    return last_checkpoint_filepath


if __name__ == "__main__":
    file_path = "/Users/yaweili/Downloads/log_final_validate.csv"
    parse_csv(file_path)
