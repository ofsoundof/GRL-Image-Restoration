# -*- coding: utf-8 -*-
import os

import random

import cv2
import numpy as np
import scipy
import scipy.io as io
import scipy.stats as ss
import torch
from scipy import ndimage
from scipy.interpolate import interp2d

from scipy.ndimage import filters, interpolation, measurements

from utils import utils_deblur, utils_image as util


"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
#
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# 28/Nov/2019
# --------------------------------------------
"""


def comp_upto_shift(I1, I2, maxshift=5, border=15, min_interval=0.25):
    """
    Args:
        I1: estimated image
        I2: reference
        maxshift: assumed maxshift
        border: shave border to calculate PSNR and SSIM
    Returns:
        PSNR and SSIM
    """

    I2 = I2[border:-border, border:-border]
    I1 = I1[
        border - maxshift : -border + maxshift, border - maxshift : -border + maxshift
    ]
    N1, N2 = I2.shape[:2]

    gx, gy = np.arange(-maxshift, N2 + maxshift, 1.0), np.arange(
        -maxshift, N1 + maxshift, 1.0
    )

    shifts = np.linspace(-maxshift, maxshift, int(2 * maxshift / min_interval + 1))
    gx0, gy0 = np.arange(0, N2, 1.0), np.arange(0, N1, 1.0)

    ssdem = np.zeros([len(shifts), len(shifts)])
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            gxn = gx0 + shifts[i]
            gvn = gy0 + shifts[j]
            if I1.ndim == 2:
                tI1 = interp2d(gx, gy, I1)(gxn, gvn)
            elif I1.ndim == 3:
                tI1 = np.zeros(I2.shape)
                for k in range(I1.shape[-1]):
                    tI1[:, :, k] = interp2d(gx, gy, I1[:, :, k])(gxn, gvn)
            ssdem[i, j] = np.sum((tI1 - I2) ** 2)

    # util.surf(ssdem)
    idxs = np.unravel_index(np.argmin(ssdem), ssdem.shape)
    print("shifted pixel is {}x{}".format(shifts[idxs[0]], shifts[idxs[1]]))

    gxn = gx0 + shifts[idxs[0]]
    gvn = gy0 + shifts[idxs[1]]
    if I1.ndim == 2:
        tI1 = interp2d(gx, gy, I1)(gxn, gvn)
    elif I1.ndim == 3:
        tI1 = np.zeros(I2.shape)
        for k in range(I1.shape[-1]):
            tI1[:, :, k] = interp2d(gx, gy, I1[:, :, k])(gxn, gvn)
    psnr = util.calculate_psnr(tI1, I2, border=0)
    ssim = util.calculate_ssim(tI1, I2, border=0)
    return psnr, ssim


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
        np.array([1.0, 0.0]),
    )
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


"""
# --------------------------------------------
# calculate PCA projection matrix
# --------------------------------------------
"""


def get_pca_matrix(x, dim_pca=15):
    """
    Args:
        x: 225x10000 matrix
        dim_pca: 15

    Returns:
        pca_matrix: 15x225
    """
    C = np.dot(x, x.T)
    w, v = scipy.linalg.eigh(C)
    pca_matrix = v[:, -dim_pca:].T

    return pca_matrix


def show_pca(x):
    """
    x: PCA projection matrix, e.g., 15x225
    """
    for i in range(x.shape[0]):
        xc = np.reshape(x[i, :], (int(np.sqrt(x.shape[1])), -1), order="F")
        util.surf(xc)


def cal_pca_matrix(
    path="PCA_matrix.mat", ksize=15, l_max=12.0, dim_pca=15, num_samples=500
):
    kernels = np.zeros([ksize * ksize, num_samples], dtype=np.float32)
    for i in range(num_samples):

        theta = np.pi * np.random.rand(1)
        l1 = 0.1 + l_max * np.random.rand(1)
        l2 = 0.1 + (l1 - 0.1) * np.random.rand(1)

        k = anisotropic_Gaussian(ksize=ksize, theta=theta[0], l1=l1[0], l2=l2[0])

        # util.imshow(k)

        kernels[:, i] = np.reshape(k, (-1), order="F")  # k.flatten(order='F')

    # io.savemat('k.mat', {'k': kernels})

    pca_matrix = get_pca_matrix(kernels, dim_pca=dim_pca)

    io.savemat(path, {"p": pca_matrix})

    return pca_matrix


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def blur(x, k):
    """
    x: image, NxcxHxW
    k: kernel, Nx1xhxw
    """
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode="replicate")
    k = k.repeat(1, c, 1, 1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n * c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x


"""
# --------------------------------------------
# kernel shift
# --------------------------------------------
"""


def gen_kernel(
    k_size=np.array([15, 15]),
    scale_factor=np.array([4, 4]),
    min_var=0.6,
    max_var=10.0,
    noise_level=0,
):
    """ "
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5 * (scale_factor - 1)  # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calculate Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    # raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def bicubic_degradation(x, sf=3):
    """
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor

    Return:
        bicubicly downsampled LR image
    """
    x = util.imresize_np(x, scale=1 / sf)
    return x


def srmd_degradation(x, k, sf=3):
    """blur + bicubic downsampling

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2018learning,
          title={Learning a single convolutional super-resolution network for multiple degradations},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={3262--3271},
          year={2018}
        }
    """
    x = ndimage.filters.convolve(
        x, np.expand_dims(k, axis=2), mode="wrap"
    )  # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)
    return x


def dpsr_degradation(x, k, sf=3):

    """bicubic downsampling + blur

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2019deep,
          title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1671--1681},
          year={2019}
        }
    """
    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode="wrap")
    return x


def classical_degradation(x, k, sf=3):
    """blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    """
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode="wrap")
    # x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def degradation_sr(img, sf, ispmodel):

    lsts = random.sample(range(7), 7)
    idx1, idx2 = lsts.index(1), lsts.index(2)
    if idx1 > idx2:
        lsts[idx1], lsts[idx2] = lsts[idx2], lsts[idx1]

    for ii in lsts:

        if ii == 0:
            # blur
            if np.random.rand() < 0.5:
                l1 = np.random.uniform(0.1, 6)
                l2 = np.random.uniform(0.1, l1)
                k = anisotropic_Gaussian(
                    ksize=11, theta=np.random.uniform(0.0, 1.0) * np.pi, l1=l1, l2=l2
                )
            else:
                k = utils_deblur.fspecial("gaussian", 11, 1.4 * np.random.uniform(0, 1))
            if np.random.rand() < 0.5:
                img = ndimage.filters.convolve(
                    img, np.expand_dims(k, axis=2), mode="wrap"
                )
            else:
                img = ndimage.filters.convolve(
                    img, np.expand_dims(k, axis=2), mode="mirror"
                )

        elif ii == 1:
            # downsample 1
            a, b = img.shape[1], img.shape[0]
            if np.random.rand() > 0.2:
                sf1, sf2 = random.choice([1, 2, 4]), random.choice([1, 2, 4])
                img = cv2.resize(
                    img,
                    (int(1 / sf1 * img.shape[1]), int(1 / sf2 * img.shape[0])),
                    interpolation=np.random.randint(1, 6),
                )

        elif ii == 2:
            # downsample 2
            if np.random.rand() > 0.5:
                img = cv2.resize(
                    img,
                    (int(1 / sf * a), int(1 / sf * b)),
                    interpolation=np.random.randint(1, 6),
                )
            else:
                k = utils_deblur.fspecial("gaussian", 11, 1.4 * np.random.uniform(0, 1))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()
                if np.random.rand() < 0.5:
                    img = ndimage.filters.convolve(
                        img, np.expand_dims(k, axis=2), mode="wrap"
                    )
                else:
                    img = ndimage.filters.convolve(
                        img, np.expand_dims(k, axis=2), mode="mirror"
                    )
                img = cv2.resize(
                    img, (int(1 / sf * a), int(1 / sf * b)), interpolation=0
                )

        elif ii == 3:
            # add noise 1
            noise_level = np.random.randint(1, 16)
            if np.random.rand() > 0.5:
                img += np.random.normal(0, noise_level / 255.0, img.shape).astype(
                    np.float32
                )
            else:
                img += np.random.normal(
                    0, noise_level / 255.0, (*img.shape[:2], 1)
                ).astype(np.float32)
            img = util.uint2single(util.single2uint(img))

        elif ii == 4:
            if np.random.rand() > 0.5:
                with torch.no_grad():
                    img = ispmodel.forward(img.copy())

        elif ii == 5:
            if np.random.rand() > 0.2:
                quality_factor = np.random.randint(30, 96)
                img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
                result, encimg = cv2.imencode(
                    ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
                )
                img = cv2.imdecode(encimg, 1)
                img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)

        elif ii == 6:
            k = utils_deblur.fspecial("gaussian", 11, 1.4 * np.random.uniform(0, 1))
            img = ndimage.filters.convolve(
                img, np.expand_dims(k, axis=2), mode="mirror"
            )

    #            r_num = np.random.rand()
    #            if  r_num < 0.1:
    #                sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    #                img = cv2.filter2D(util.single2uint(img), cv2.CV_32F, sharpen_k)
    #                img = util.uint2single(cv2.convertScaleAbs(img))
    #            elif r_num > 0.8:
    #                radius = 0.9+0.2*np.random.uniform(0, 1)
    #                threshold = np.random.uniform(0, 20)
    #                amount = 2.0*np.random.uniform(0, 1) # (0, 2)
    #                sharpen_k = utils_deblur.fspecial('gaussian', max(3, int(np.ceil(radius)*2+1)), radius).astype(np.float32)
    #                ycbcr = np.matmul(img*255., [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
    #                                  [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    #                y = ycbcr[...,0]
    #                y_lf = ndimage.filters.convolve(y, sharpen_k, mode='mirror')
    #                y_hf = y-y_lf
    #                imLabel = (y_hf > threshold)
    #                ycbcr[...,0] = y + amount * y_hf * imLabel
    #                img = np.matmul(ycbcr, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
    #                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    #                img = img/255.
    #                img = np.float32(img)

    # add JPEG compression noise, 1
    quality_factor = np.random.randint(30, 96)
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    )
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)

    # img = util.uint2single(util.single2uint(img))

    return img


# def degradation_md_noise_jpeg(img, scale_ratio, sigma, quality_factor, noise_level):
#    # get shifted Gausssian kernel
#    k = utils_deblur.fspecial('gaussian', 15, sigma)
#    k_shifted = shift_pixel(k, scale_ratio)
#    k_shifted = k_shifted/k_shifted.sum()
#
#    # get clean LR
#    img_L_clean = classical_degradation(img, k_shifted, sf=scale_ratio)
#
#    # add Gaussian noise
# 	if np.random.rand() > 0.5:
#        if np.random.rand() > 0.5:
#            img_L_clean += np.random.normal(0, noise_level/255.0, img_L_clean.shape)
#        else:
#            img_L_clean += np.random.normal(0, noise_level/255.0, (*img_L_clean.shape[:2],1))
#
#    # add JPEG compression noise
#    img_L_clean = cv2.cvtColor(util.single2uint(img_L_clean), cv2.COLOR_RGB2BGR)
#    result, encimg = cv2.imencode('.jpg', img_L_clean, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
#    img_L = cv2.imdecode(encimg, 1)
#    img_lr = cv2.cvtColor(util.uint2single(img_L), cv2.COLOR_BGR2RGB)
#
#    return img_lr


def degradation_bic_jpeg(img, scale_ratio, sigma, quality_factor):
    # get shifted Gausssian kernel

    # get clean LR
    img_L_clean = bicubic_degradation(img, sf=scale_ratio)

    # add JPEG compression noise
    img_L_clean = cv2.cvtColor(util.single2uint(img_L_clean), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode(
        ".jpg", img_L_clean, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    )
    img_L = cv2.imdecode(encimg, 1)
    img_lr = cv2.cvtColor(util.uint2single(img_L), cv2.COLOR_BGR2RGB)

    return img_lr


def modcrop_np(img, sf):
    """
    Args:
        img: numpy image, WxH or WxHxC
        sf: scale factor

    Return:
        cropped image
    """
    w, h = img.shape[:2]
    im = np.copy(img)
    return im[: w - w % sf, : h - h % sf, ...]


if __name__ == "__main__":
    img = util.imread_uint("utils/test.bmp", 3)
    img = util.uint2single(img)

    img = util.single2tensor4(img)
    img = img.repeat(6, 1, 1, 1)
    print(img.shape)

    weight = utils_deblur.fspecial("gaussian", 9, 9.1)[:, :, None]
    weight = util.single2tensor4(weight).repeat(6, 1, 1, 1)  # Nx3
    img1 = blur(img, weight)

    util.imshow(util.tensor2uint(img1[5, ...]))

    img = img.view(1, -1, 256, 256)  # 1x(Nx3)
    print(img.shape)
    util.imshow(util.tensor2uint(img[:, [0, 1, 2], :, :]))

    from utils import utils_deblur

    weight = utils_deblur.fspecial("gaussian", 9, 2)[:, :, None]
    weight1 = utils_deblur.fspecial("gaussian", 9, 0.1)[:, :, None]
    weight1 = util.single2tensor4(weight1).repeat(1, 3, 1, 1)

    weight = util.single2tensor4(weight).repeat(6, 3, 1, 1)  # Nx3
    weight[1, :, :, :] = weight1[0, :, :, :]
    print(weight.shape)
    weight = weight.view(-1, 1, 9, 9)  # (Nx3)x1

    img1 = torch.nn.functional.conv2d(
        img, weight, bias=None, stride=1, padding=4, dilation=1, groups=18
    )
    print(img1.shape)

    img1 = img1.view(6, 3, 256, 256)

    util.imshow(util.tensor2uint(img1[0, :, :, :]))
    util.imshow(util.tensor2uint(img1[1, :, :, :]))

    from utils import utils_isp

    ispmodel = utils_isp.ISPModel()

    #    radius = 1 #0.9+0.2*np.random.uniform(0, 1)
    #    threshold = 0. #np.random.uniform(0, 100)
    #    amount = 2 #10.921.6*np.random.uniform(0, 1) # (0, 2)
    #    sharpen_k = utils_deblur.fspecial('gaussian', max(3, int(np.ceil(radius)*2+1)), radius).astype(np.float32)
    #    ycbcr = np.matmul(img*255., [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
    #                      [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    #    y = ycbcr[...,0]
    #    y_lf = ndimage.filters.convolve(y, sharpen_k, mode='mirror')
    #    y_hf = y-y_lf
    #
    #    print(np.max(y_hf))
    #
    #    imLabel = (y_hf > threshold)
    #    ycbcr[...,0] = y + amount * y_hf * imLabel
    #    img = np.matmul(ycbcr, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
    #                  [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    #    img = img/255.
    #
    #    util.imsave(util.single2uint(img), '111.png')

    #
    #    sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    #    img = cv2.filter2D(util.single2uint(img), cv2.CV_32F, sharpen_op)
    #
    #    img = cv2.convertScaleAbs(img)
    #
    #    util.imsave(img, '222.png')
    #
    #    img = util.uint2single(img)

    # k = gen_kernel(scale_factor=np.array([1, 1]))
    l1 = np.random.uniform(0.1, 8)
    l2 = np.random.uniform(0.1, l1)
    k = anisotropic_Gaussian(
        ksize=15, theta=np.random.uniform(0.0, 1.0) * np.pi, l1=l1, l2=l2
    )
    util.imshow(k)

    for i in range(100):
        img1 = degradation_sr(img, 2, ispmodel)
        util.imshow(util.single2uint(img1))
        util.imsave(util.single2uint(img1), str(i) + ".png")

    # run utils/utils_sisr.py

#    img = util.uint2single(img)
#    k = utils_deblur.fspecial('gaussian', 7, 1.6)
#
#    for sf in [2, 3, 4]:
#
#        # modcrop
#        img = modcrop_np(img, sf=sf)
#
#        # 1) bicubic degradation
#        img_b = bicubic_degradation(img, sf=sf)
#        print(img_b.shape)
#
#        # 2) srmd degradation
#        img_s = srmd_degradation(img, k, sf=sf)
#        print(img_s.shape)
#
#        # 3) dpsr degradation
#        img_d = dpsr_degradation(img, k, sf=sf)
#        print(img_d.shape)

#
#    sf = 4
#
#    # modcrop
#    img = modcrop_np(img, sf=sf)
#
#    # shifted Gaussian blur kernel
#
#    for sigma in np.linspace(0.4, 2.4, 6):
#        print(sigma)
#        k = utils_deblur.fspecial('gaussian', 15, sigma)
##        print(k)
#        util.imshow(k*10)
#        k_shifted = shift_pixel(k, sf)
#
##        print(sum())
#
#
##        print(k_shifted)
#        util.imshow(k_shifted*10)
#        util.surf(k_shifted*10)
#        # get clean LR
#        img_L = classical_degradation(img, k_shifted, sf=sf)
#        util.imshow(img_L)
#
#        # add JPEG compression noise
#        quality_factor = 75
#        cv2.imwrite('tmp.jpg', cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
#        img_L_compressed_image = cv2.cvtColor(cv2.imread('tmp.jpg'), cv2.COLOR_BGR2RGB)
#   # os.remove('tmp.jpg')
#
#
#
#
#
#
#        k = anisotropic_Gaussian(ksize=15, theta=0.25*np.pi, l1=0.01, l2=0.01)
#        util.imshow(k*10)
#        k = gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.8, max_var=10.8, noise_level=0.0)
#        util.imshow(k*10)

#    util.surf(k)

# PCA
#    pca_matrix = cal_pca_matrix(ksize=15, l_max=10.0, dim_pca=15, num_samples=12500)
#    print(pca_matrix.shape)
#    show_pca(pca_matrix)
# run utils/utils_sisr.py
