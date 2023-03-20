"""
Created on Thu Jan 18 15:36:32 2018
@author: italo
https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
"""

"""
Syntax
h = fspecial(type)
h = fspecial('average',hsize)
h = fspecial('disk',radius)
h = fspecial('gaussian',hsize,sigma)
h = fspecial('laplacian',alpha)
h = fspecial('log',hsize,sigma)
h = fspecial('motion',len,theta)
h = fspecial('prewitt')
h = fspecial('sobel')
"""

import os.path as osp

import numpy as np
import scipy
import torch


def fspecial_average(hsize=3):
    """Smoothing filter"""
    return np.ones((hsize, hsize)) / hsize**2


def fspecial_disk(radius):
    """Disk filter"""
    raise NotImplementedError
    rad = 0.6
    crad = np.ceil(rad - 0.5)
    [x, y] = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.zeros(x.shape)
    maxxy[abs(x) >= abs(y)] = abs(x)[abs(x) >= abs(y)]
    maxxy[abs(y) >= abs(x)] = abs(y)[abs(y) >= abs(x)]
    minxy = np.zeros(x.shape)
    minxy[abs(x) <= abs(y)] = abs(x)[abs(x) <= abs(y)]
    minxy[abs(y) <= abs(x)] = abs(y)[abs(y) <= abs(x)]
    m1 = (rad**2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + (
        rad**2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2
    ) * np.sqrt((rad**2 + 0j) - (maxxy + 0.5) ** 2)
    m2 = (rad**2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + (
        rad**2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2
    ) * np.sqrt((rad**2 + 0j) - (maxxy - 0.5) ** 2)
    h = None
    return h


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial_log(hsize, sigma):
    raise NotImplementedError


def fspecial_motion(motion_len, theta):
    raise NotImplementedError


def fspecial_prewitt():
    return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


def fspecial_sobel():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def fspecial(filter_type, *args, **kwargs):
    """
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    """
    if filter_type == "average":
        return fspecial_average(*args, **kwargs)
    if filter_type == "disk":
        return fspecial_disk(*args, **kwargs)
    if filter_type == "gaussian":
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == "laplacian":
        return fspecial_laplacian(*args, **kwargs)
    if filter_type == "log":
        return fspecial_log(*args, **kwargs)
    if filter_type == "motion":
        return fspecial_motion(*args, **kwargs)
    if filter_type == "prewitt":
        return fspecial_prewitt(*args, **kwargs)
    if filter_type == "sobel":
        return fspecial_sobel(*args, **kwargs)


def get_blur_kernel(kernel_type):
    KERNEL_DIR = osp.join(
        osp.dirname(osp.abspath(__file__)), "blur_kernels/Levin09.npy"
    )

    if kernel_type == "gaussian":
        kernel = fspecial("gaussian", 25, 1.6)
    elif kernel_type.find("real") >= 0:
        with open(KERNEL_DIR, "rb") as f:
            kernel = np.load(f, allow_pickle=True)[0, int(kernel_type[-1]) - 1]

    kernel = torch.from_numpy(np.flip(kernel.astype(np.float32)).copy())
    kernel = kernel.repeat(3, 1, 1, 1)
    return kernel
