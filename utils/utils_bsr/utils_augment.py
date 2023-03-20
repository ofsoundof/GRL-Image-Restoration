import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from PIL import Image


plt.rcParams["savefig.bbox"] = "tight"
orig_img = Image.open("test.bmp")
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
# torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


import utils_image as util

orig_img = util.imread_uint("test.bmp", 3)
orig_img = util.uint2tensor4(orig_img)

jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

jitter(orig_img).shape


util.imshow(util.tensor2uint(jitter(orig_img)))


#
# jitted_imgs = [jitter(orig_img) for _ in range(3)]
#
# jitted_imgs = [util.tensor2uint(orig_img) for orig_img in jitted_imgs]
#
#
# plot(jitted_imgs)
