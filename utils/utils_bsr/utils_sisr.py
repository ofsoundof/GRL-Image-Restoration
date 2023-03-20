# -*- coding: utf-8 -*-
import random

import cv2
import numpy as np
import scipy
import scipy.io as io
import scipy.stats as ss
import torch
from utils.utils_bsr import utils_image as util
from scipy import ndimage
from scipy.interpolate import interp2d
from scipy.linalg import orth


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r : 2 * r + k_size, 2 * c : 2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


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


def gen_kernel(
    k_size=np.array([15, 15]),
    scale_factor=np.array([4, 4]),
    min_var=0.6,
    max_var=10.0,
    noise_level=0,
):
    """ "
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
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


def fspecial(filter_type, *args, **kwargs):
    """
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    """
    if filter_type == "gaussian":
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == "laplacian":
        return fspecial_laplacian(*args, **kwargs)


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


def degradation_sr2(img, sf, ispmodel):

    HR = img.copy()

    if sf == 4 and random.random() < 0.1:
        if np.random.rand() < 0.5:
            img = cv2.resize(
                img,
                (int(1 / 2 * img.shape[1]), int(1 / 2 * img.shape[0])),
                interpolation=random.choice([1, 2, 3]),
            )
        else:
            img = util.imresize_np(img, 1 / 2, True)
        img = np.clip(img, 0.0, 1.0)
        sf = 2

    lsts = random.sample(range(9), 9)
    idx1, idx2 = lsts.index(1), lsts.index(6)
    if idx1 > idx2:
        lsts[idx1], lsts[idx2] = lsts[idx2], lsts[idx1]

    #    if sf==4:  # (2.4 -- 6.0) (2.8--8.0)  (3.2--10.0)
    #        wd = 2.8
    #        wd2 = 8.0
    #    elif sf==2:
    #        wd = 2.4
    #        wd2 = 6.0
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf

    for ii in lsts:

        if ii == 0:
            # add blur 1
            if random.random() < 0.5:
                l1 = wd2 * random.random()
                l2 = wd2 * random.random()
                k = anisotropic_Gaussian(
                    ksize=2 * random.randint(2, 11) + 3,
                    theta=random.random() * np.pi,
                    l1=l1,
                    l2=l2,
                )
            else:
                k = fspecial(
                    "gaussian", 2 * random.randint(2, 11) + 3, wd * random.random()
                )
            # print(img.shape, k.shape)
            img = ndimage.filters.convolve(
                img, np.expand_dims(k, axis=2), mode="mirror"
            )

        elif ii == 1:
            a, b = img.shape[1], img.shape[0]
            # downsample 1
            if random.random() < 0.5:
                sf1 = random.uniform(1, 2 * sf)
                img = cv2.resize(
                    img,
                    (int(1 / sf1 * img.shape[1]), int(1 / sf1 * img.shape[0])),
                    interpolation=random.choice([1, 2, 3]),
                )
            else:
                k = fspecial("gaussian", 25, random.uniform(0.1, 0.4 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()
                img = ndimage.filters.convolve(
                    img, np.expand_dims(k_shifted, axis=2), mode="mirror"
                )
                img = img[0::sf, 0::sf, ...]
            img = np.clip(img, 0.0, 1.0)

        elif ii == 2:
            # add camera noise
            if random.random() > 0.75:
                with torch.no_grad():
                    img, HR = ispmodel.forward(img.copy(), HR)
                    # print(img.shape[0])

        elif ii == 3:
            # add Gaussian noise 1
            noise_level = random.randint(2, 25)
            rnum = np.random.rand()
            if rnum > 0.5:
                img += np.random.normal(0, noise_level / 255.0, img.shape).astype(
                    np.float32
                )
            elif rnum < 0.4:
                img += np.random.normal(
                    0, noise_level / 255.0, (*img.shape[:2], 1)
                ).astype(np.float32)
            else:
                L = 25 / 255.0
                D = np.diag(np.random.rand(3))
                U = orth(np.random.rand(3, 3))
                conv = np.dot(np.dot(np.transpose(U), D), U)
                img += np.random.multivariate_normal(
                    [0, 0, 0], np.abs(L**2 * conv), img.shape[:2]
                ).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

        elif ii == 4:
            # add JPEG noise
            if random.random() < 0.9:
                quality_factor = random.randint(20, 95)
                img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
                result, encimg = cv2.imencode(
                    ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
                )
                img = cv2.imdecode(encimg, 1)
                img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)

        elif ii == 5:
            # add blur 2
            k = fspecial(
                "gaussian", 2 * random.randint(2, 11) + 3, wd * random.random()
            )
            img = ndimage.filters.convolve(
                img, np.expand_dims(k, axis=2), mode="mirror"
            )

        elif ii == 6:
            # downsample 2
            img = cv2.resize(
                img,
                (int(1 / sf * a), int(1 / sf * b)),
                interpolation=random.choice([1, 2, 3]),
            )
            img = np.clip(img, 0.0, 1.0)

        elif ii == 7:
            noise_level = np.random.randint(2, 25)
            # add speckle noise 2
            if random.random() > 0.5:
                img = np.clip(img, 0.0, 1.0)
                rnum = random.random()
                if rnum > 0.6:
                    img += img * np.random.normal(
                        0, noise_level / 255.0, img.shape
                    ).astype(np.float32)
                elif rnum < 0.4:
                    img += img * np.random.normal(
                        0, noise_level / 255.0, (*img.shape[:2], 1)
                    ).astype(np.float32)
                else:
                    L = 25 / 255.0
                    D = np.diag(np.random.rand(3))
                    U = orth(np.random.rand(3, 3))
                    conv = np.dot(np.dot(np.transpose(U), D), U)
                    img += img * np.random.multivariate_normal(
                        [0, 0, 0], np.abs(L**2 * conv), img.shape[:2]
                    ).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)

        elif ii == 8:
            # add poisson noise 3
            if random.random() > 0.5:
                img = np.clip(img, 0.0, 1.0)
                vals = 10 ** (2 * random.random() + 2.0)  # [2, 4]
                img = np.random.poisson(img * vals).astype(np.float32) / vals
                img = np.clip(img, 0.0, 1.0)

    # add final JPEG compression noise
    quality_factor = random.randint(20, 95)
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode(
        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    )
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)

    return img, HR


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
    img = util.imread_uint("utils/b.png", 3)
    # img = util.imread_uint('utils/im_13.jpg', 3)
    img = util.uint2single(img)
    #    img = modcrop_np(img, sf=2)
    #    sf = 2
    #    a, b = img.shape[1], img.shape[0]
    #
    #  #  img_bic = cv2.resize(img, (int(1/sf*a), int(1/sf*b)), interpolation=1)
    #    img_bic = util.imresize_np(img, 1/sf, True)
    #
    #
    #    for i in range(24):
    #        k = fspecial('gaussian', 25, (i+1)/10)
    #      #  print()
    #       # util.imshow(k)
    #        k_shifted = shift_pixel(k, sf)
    #        k_shifted = k_shifted/k_shifted.sum()
    #        img1 = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
    #        img1 = img1[0::sf, 0::sf, ...]
    #        adiff = np.sum(np.abs(img_bic-img1))
    #
    #        #util.imshow(np.concatenate([util.single2uint(img_bic), util.single2uint(img1)]))
    #
    #        util.imwrite(np.concatenate([util.single2uint(img_bic), util.single2uint(img1)]),str(i)+'.png')
    #
    #
    #        print([(i+1)/10,adiff])

    # run utils/utils_sisr.py

    # sf = 2 [0.1, 0.8]
    # sf = 3 [0.1, 1.2]
    # sf = 4 [0.1, 1.6]

    #  util.imshow(util.augment_img(img,4))

    #    img += np.random.normal(0, 5/255.0, (*img.shape[:2], 1)).astype(np.float32)
    #
    #    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    #    img = cv2.filter2D(util.single2uint(img), cv2.CV_32F, sharpen_k)
    #    img = util.uint2single(cv2.convertScaleAbs(img))

    #    radius = 0.9+0.2*np.random.uniform(0, 1)
    #    threshold = np.random.uniform(0, 20)
    #    amount = 2.0*np.random.uniform(0, 1) # (0, 2)
    #    sharpen_k = utils_deblur.fspecial('gaussian', max(3, int(np.ceil(radius)*2+1)), radius).astype(np.float32)
    #    ycbcr = np.matmul(img*255., [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
    #                      [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    #    y = ycbcr[...,0]
    #    y_lf = ndimage.filters.convolve(y, sharpen_k, mode='mirror')
    #    y_hf = y-y_lf
    #    imLabel = (y_hf > threshold)
    #    ycbcr[...,0] = y + amount * y_hf * imLabel
    #    img = np.matmul(ycbcr, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
    #                  [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    #    img = img/255.
    #    img = np.float32(img)

    #    util.imsave(util.single2uint(img),'aa.png')

    #    img =  util.single2tensor4(img)
    #    img = img.repeat(6,1,1,1)
    #    print(img.shape)
    #
    #
    #    weight = utils_deblur.fspecial('gaussian',9, 9.1)[:,:,None]
    #    weight =  util.single2tensor4(weight).repeat(6,1,1,1)  # Nx3
    #    img1 = blur(img, weight)
    #
    #    util.imshow(util.tensor2uint(img1[5,...]))

    #    img = img.view(1,-1,256,256)  # 1x(Nx3)
    #    print(img.shape)
    #    util.imshow(util.tensor2uint(img[:,[0,1,2],:,:]))
    #
    #    from utils import utils_deblur
    #    weight = utils_deblur.fspecial('gaussian',9, 2)[:,:,None]
    #    weight1 = utils_deblur.fspecial('gaussian',9, 0.1)[:,:,None]
    #    weight1 =  util.single2tensor4(weight1).repeat(1,3,1,1)
    #
    #
    #    weight =  util.single2tensor4(weight).repeat(6,3,1,1)  # Nx3
    #    weight[1,:,:,:] = weight1[0,:,:,:]
    #    print(weight.shape)
    #    weight = weight.view(-1,1,9,9)  # (Nx3)x1
    #
    #    img1 = torch.nn.functional.conv2d(img, weight, bias=None, stride=1, padding=4, dilation=1, groups=18)
    #    print(img1.shape)
    #
    #    img1 = img1.view(6,3,256,256)
    #
    #    util.imshow(util.tensor2uint(img1[0,:,:,:]))
    #    util.imshow(util.tensor2uint(img1[1,:,:,:]))

    from utils import utils_isp

    ispmodel = utils_isp.ISPModel()
    #    tgmodel = utils_isp.TGModel()

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

    #    #k = gen_kernel(scale_factor=np.array([1, 1]))
    #    l1 = np.random.uniform(0.1, 8)
    #    l2 = np.random.uniform(0.1, l1)
    #    k = anisotropic_Gaussian(ksize=15, theta=np.random.uniform(0.0, 1.0)*np.pi, l1=l1, l2=l2)
    #    util.imshow(k)
    #
    print(img.shape)
    for i in range(100):
        img1, _ = degradation_sr2(img, 4, ispmodel)
        print(i)
        #   print(img1.shape)
        #        util.imshow(util.single2uint(img1))
        util.imsave(
            cv2.resize(
                util.single2uint(img1),
                (int(8 * img1.shape[1]), int(8 * img1.shape[0])),
                interpolation=0,
            ),
            str(i) + ".png",
        )

#    run utils/utils_sisr.py


# k0 = anisotropic_Gaussian(ksize=21, theta=np.random.uniform(0, 1.0)*np.pi, l1=8, l2=8)
# k1 = utils_deblur.fspecial('gaussian', 21, 2.82859)
# sum(sum(abs(k0-k1)))
#
#
# k0 = anisotropic_Gaussian(ksize=21, theta=np.random.uniform(0, 1.0)*np.pi, l1=4, l2=4)
# k1 = utils_deblur.fspecial('gaussian', 21, 1.6)
# sum(sum(abs(k0-k1)))
#
#
# util.imshow(utils_deblur.fspecial('gaussian', 21, 2.8))
#


# print(k.sum())
# util.surf(anisotropic_Gaussian(ksize=11, theta=np.random.uniform(0, 1.0)*np.pi, l1=10, l2=10))
#
# util.surf(utils_deblur.fspecial('gaussian', 21, 3.2))

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
