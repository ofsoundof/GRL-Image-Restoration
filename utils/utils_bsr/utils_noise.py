"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import os

import cv2

# import numpy as np
import numpy as np

# We adopt the same noise sampling procedure as in "Unprocessing Images for Learned Raw Denoising" by Brooks et al. CVPR2019

import torch
import torch.distributions as dist


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


# If the target dataset is DND, use this function #####################
def random_noise_levels_dnd():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    # log_max_shot_noise = torch.log10(torch.Tensor([0.012]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.006]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10, log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.25]))
    read_noise = distribution.sample().clamp(min=-1.5, max=1.5)
    # line = lambda x: 2.18 * x + 1.20
    line = lambda x: 2.275 * x + 1.47
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10, log_read_noise)
    return shot_noise, read_noise


# If the target dataset is SIDD, use this function #####################
def random_noise_levels_sidd():
    """Where read_noise in SIDD is not 0"""
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    # log_max_shot_noise = torch.log10(torch.Tensor([0.022]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.010]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)
    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10, log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.20]))
    read_noise = distribution.sample().clamp(min=-1.12, max=1.12)
    # line = lambda x: 1.85 * x + 0.30  # Line SIDD test set
    line = lambda x: 1.84 * x + 0.27  # Line SIDD test set
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10, log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005, use_cuda=False):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    mean = torch.Tensor([0.0])
    if use_cuda:
        mean = mean.cuda()
    distribution = dist.normal.Normal(mean, torch.sqrt(variance))
    noise = distribution.sample()
    return image + noise


def add_rawnoise(image):
    """Adds random shot (proportional to image) and read (independent) noise."""

    if torch.rand(1) > 0.5:
        shot_noise, read_noise = random_noise_levels_sidd()
    else:
        shot_noise, read_noise = random_noise_levels_dnd()

    variance = image * shot_noise + read_noise
    #    if torch.rand(1)>0.3:
    #        variance = image * shot_noise + read_noise
    #    else:
    #        variance = torch.sum(torch.FloatTensor([0.299,0.587,0.114]).type_as(image).view(1,3,1,1)*image,1,True) * shot_noise + read_noise

    mean = torch.Tensor([0.0])
    distribution = dist.normal.Normal(mean, torch.sqrt(variance))
    noise = distribution.sample()
    return image + noise


if __name__ == "__main__":
    from utils import utils_image as util

    #    print(random_noise_levels_dnd())
    # run utils/utils_noise.py
    img = util.imread_uint("utils/test.bmp", 3)
    #    img = util.imread_uint('utils/b.png', 3)
    img = util.uint2single(img)

    noise_level = 25

    #    scale = power(10, noiseLevel);
    #    noise = scale * imnoise(im2double(imgVec{j})/scale, 'poisson');

    #    scale = 10**0.1
    #    noise = scale *np.random.poisson(img/scale)

    #    vals = len(np.unique(img))
    #    vals = 2 ** np.ceil(np.log2(vals))
    #    vals = 10**2  # [2, 4]
    #    img = np.random.poisson(img * vals).astype(np.float32) / vals
    #    img = util.single2uint(img)
    #    util.imsave(img,'spec_noisy.png')
    #
    from scipy.linalg import orth

    L = 25 / 255
    D = np.diag(np.random.rand(3))
    U = orth(np.random.rand(3, 3))
    conv = np.dot(np.dot(np.transpose(U), D), U)
    #    conv = np.ones((3,3))
    imageNoiseSigma = np.abs(L**2 * conv)
    img += np.random.multivariate_normal(
        [0, 0, 0], imageNoiseSigma, img.shape[:2]
    ).astype(np.float32)
    img = util.single2uint(img)
    util.imsave(img, "spec_noisy.png")


#
#    #img = img + img * np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
#    img = img + img * np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
#    img = util.single2uint(img)
#    util.imsave(img,'spec_noisy.png')
#
#    gauss = np.random.randn(row,col,ch)
#    gauss = gauss.reshape(row,col,ch)
#    noisy = image + image * gauss
