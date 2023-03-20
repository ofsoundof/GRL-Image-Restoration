# coding=utf-8

# import rawpy
import os.path
import random

import numpy as np
import torch
import torch.nn as nn
from utils.utils_bsr import (
    utils_color,
    utils_image as util,
    utils_mat as mat,
    utils_noise,
)

# import hdf5storage

from scipy.interpolate import interp1d

CAMERA_PROFILE_DIR = os.path.join(
    os.path.abspath(__file__).split("utils_bsr")[0], "cameraprofile"
)


class ColorCorrection_conv1x1(nn.Module):
    def __init__(self, weight):
        super(ColorCorrection_conv1x1, self).__init__()
        self.weight = weight
        self.weight_inv = torch.inverse(self.weight).unsqueeze(-1).unsqueeze(-1)
        self.weight = self.weight.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        # return nn.functional.conv2d(x, self.weight, bias=None)
        return torch.sum(
            x.unsqueeze(2) * self.weight.unsqueeze(0), dim=2, keepdim=False
        )

    def reverse(self, x):
        # return nn.functional.conv2d(x, self.weight_inv, bias=None)
        return torch.sum(
            x.unsqueeze(2) * self.weight_inv.unsqueeze(0), dim=2, keepdim=False
        )


class Raw2XYZ(nn.Module):
    """
    camera raw --> XYZ(D50)
    """

    def __init__(self, weight):
        super(Raw2XYZ, self).__init__()
        self.weight = weight
        self.weight_inv = torch.inverse(self.weight).unsqueeze(-1).unsqueeze(-1)
        self.weight = self.weight.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        # return nn.functional.conv2d(x, self.weight, bias=None)
        return torch.matmul(x.permute(0, 2, 3, 1), self.weight.squeeze().t()).permute(
            0, 3, 1, 2
        )

    def reverse(self, x):
        # return nn.functional.conv2d(x, self.weight_inv, bias=None)
        return torch.matmul(
            x.permute(0, 2, 3, 1), self.weight_inv.squeeze().t()
        ).permute(0, 3, 1, 2)


class XYZ2LinearRGB(nn.Module):
    """
    XYZ(D50) --> linear RGB
    ColorSpace=0: linear sRGB(D65)
    ColorSpace=1: linear Adobe RGB-D65
    ColorSpace=2: linear ProPhoto RGB(D50)
    """

    def __init__(self, ColorSpace=0):
        super(XYZ2LinearRGB, self).__init__()
        self.ColorSpace = ColorSpace
        self.weight = utils_color.xyz2linearrgb_weight(self.ColorSpace, is_forward=True)
        self.weight_inv = torch.inverse(self.weight).unsqueeze(-1).unsqueeze(-1)
        self.weight = self.weight.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, bias=None)

    def reverse(self, x):
        return nn.functional.conv2d(x, self.weight_inv, bias=None)


class ExposureCompensation(nn.Module):
    """
    Exposure Compensation
    BaselineExposure: from camera profile
    BaselineExposureOffset: for user adjustment
    """

    def __init__(self, BaselineExposure=0, BaselineExposureOffset=0):
        super(ExposureCompensation, self).__init__()
        self.BaselineExposure = BaselineExposure
        self.BaselineExposureOffset = BaselineExposureOffset  # for rendering
        self.exposure = self.BaselineExposure + self.BaselineExposureOffset

    def forward(self, x):
        return x.mul_(2**self.exposure).clamp_(0, 1)

    def reverse(self, x):
        return x.div_(2**self.exposure).clamp_(0, 1)


class ProfileHueSatMapEncoding(nn.Module):
    """
    XYZ(D50) --> ProPhoto RGB --> HSV(Encoding) --> ProPhoto RGB
    """

    def __init__(
        self,
        ProfileHueSatMapData1=None,
        ProfileHueSatMapData2=None,
        ProfileHueSatMapDims=None,
        ProfileHueSatMapWeightFactor=1,
    ):
        super(ProfileHueSatMapEncoding, self).__init__()
        self.NeedEncoding = False if ProfileHueSatMapData1 is None else True
        if self.NeedEncoding:
            self.ProfileHueSatMapData1 = ProfileHueSatMapData1.reshape(-1, 3).t()
            self.ProfileHueSatMapData2 = ProfileHueSatMapData2.reshape(-1, 3).t()
            self.ProfileHueSatMapWeightFactor = ProfileHueSatMapWeightFactor
            print(self.ProfileHueSatMapWeightFactor)
            self.ProfileHueSatMapDims = ProfileHueSatMapDims
            self.ProfileHueSatMapData = (
                self.ProfileHueSatMapWeightFactor * self.ProfileHueSatMapData1
                + (1 - self.ProfileHueSatMapWeightFactor) * self.ProfileHueSatMapData2
            )
        self.weight = utils_color.xyz2linearrgb_weight(2, is_forward=True)
        self.weight_inv = torch.inverse(self.weight).unsqueeze(-1).unsqueeze(-1)
        self.weight = self.weight.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        x = nn.functional.conv2d(x, self.weight, bias=None)  # xyz2linear-ProPhoto-RGB
        if self.NeedEncoding:
            hsv = utils_color.rgb2hsv(x.clamp_(0, 1), True)
            hsv = utils_color.HsvLookup(
                hsv, self.ProfileHueSatMapData, self.ProfileHueSatMapDims
            )
            x = utils_color.rgb2hsv(hsv, False)
        return x.clamp_(0, 1)

    def reverse(self, x):
        # TODO
        x = nn.functional.conv2d(x, self.weight_inv, bias=None)
        return x


class ProfileLookTableEncoding(nn.Module):
    """
    Linear ProPhoto RGB --> HSV(Encoding) --> Linear ProPhoto RGB --> XYZ(D50)
    """

    def __init__(self, ProfileLookTableData=None, ProfileLookTableDims=None):
        super(ProfileLookTableEncoding, self).__init__()
        self.NeedEncoding = False if ProfileLookTableData is None else True
        if self.NeedEncoding:
            self.ProfileLookTableData = ProfileLookTableData.reshape(-1, 3).t()
            self.ProfileLookTableDims = ProfileLookTableDims
        self.weight_prophotorgb = utils_color.xyz2linearrgb_weight(2, is_forward=True)
        self.weight_prophotorgb_inv = (
            torch.inverse(self.weight_prophotorgb).unsqueeze(-1).unsqueeze(-1)
        )
        self.weight_prophotorgb = self.weight_prophotorgb.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        if self.NeedEncoding:
            hsv = utils_color.rgb2hsv(x.clamp_(0, 1), True)
            hsv = utils_color.HsvLookup(
                hsv, self.ProfileLookTableData, self.ProfileLookTableDims
            )
            x = utils_color.rgb2hsv(hsv, False)
        x = nn.functional.conv2d(
            x, self.weight_prophotorgb_inv, bias=None
        )  # linear ProPhoto RGB --> xyz(d50)
        return x.clamp_(0, 1)

    def reverse(self, x):
        x = nn.functional.conv2d(
            x, self.weight_prophotorgb, bias=None
        )  # xyz(d50) --> linear ProPhoto RGB
        # TODO
        return x.clamp_(0, 1)


class GammaCorrect(nn.Module):
    """
    Gamma correction
    linear RGB --> RGB
    ColorSpace=0: sRGB(D65)
    ColorSpace=1: Adobe RGB(D65)
    ColorSpace=2: ProPhoto RGB(D50)
    """

    def __init__(self, ColorSpace=0):
        super(GammaCorrect, self).__init__()
        self.ColorSpace = ColorSpace

    def forward(self, x):
        x = utils_color.linear2gamma(x.clamp_(0, 1), self.ColorSpace, True)
        return x.clamp_(0, 1)

    def reverse(self, x):
        x = utils_color.linear2gamma(x.clamp_(0, 1), self.ColorSpace, False)
        return x.clamp_(0, 1)


class Demosaic(nn.Module):
    """matlab demosaicking
    Args:
        x: Nx1xWxH with RGGB bayer pattern

    Returns:
        output: Nx3xWxH
    """

    def __init__(self, NeedDemosaic=True):
        super(Demosaic, self).__init__()
        self.NeedDemosaic = NeedDemosaic
        if self.NeedDemosaic:
            kgrb = (
                1
                / 8
                * torch.FloatTensor(
                    [
                        [0, 0, -1, 0, 0],
                        [0, 0, 2, 0, 0],
                        [-1, 2, 4, 2, -1],
                        [0, 0, 2, 0, 0],
                        [0, 0, -1, 0, 0],
                    ]
                )
            )
            krbg0 = (
                1
                / 8
                * torch.FloatTensor(
                    [
                        [0, 0, 1 / 2, 0, 0],
                        [0, -1, 0, -1, 0],
                        [-1, 4, 5, 4, -1],
                        [0, -1, 0, -1, 0],
                        [0, 0, 1 / 2, 0, 0],
                    ]
                )
            )
            krbg1 = krbg0.t()
            krbbr = (
                1
                / 8
                * torch.FloatTensor(
                    [
                        [0, 0, -3 / 2, 0, 0],
                        [0, 2, 0, 2, 0],
                        [-3 / 2, 0, 6, 0, -3 / 2],
                        [0, 2, 0, 2, 0],
                        [0, 0, -3 / 2, 0, 0],
                    ]
                )
            )
            self.k = torch.stack((kgrb, krbg0, krbg1, krbbr), 0).unsqueeze(1)

    def forward(self, x):
        if self.NeedDemosaic:
            output = x.repeat(1, 3, 1, 1)
            x = nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
            conv_cfa = nn.functional.conv2d(x, self.k, padding=0, bias=None)
            output[:, 1, 0::2, 0::2] = conv_cfa[:, 0, 0::2, 0::2]
            output[:, 1, 1::2, 1::2] = conv_cfa[:, 0, 1::2, 1::2]
            output[:, 0, 0::2, 1::2] = conv_cfa[:, 1, 0::2, 1::2]
            output[:, 0, 1::2, 0::2] = conv_cfa[:, 2, 1::2, 0::2]
            output[:, 0, 1::2, 1::2] = conv_cfa[:, 3, 1::2, 1::2]
            output[:, 2, 0::2, 1::2] = conv_cfa[:, 2, 0::2, 1::2]
            output[:, 2, 1::2, 0::2] = conv_cfa[:, 1, 1::2, 0::2]
            output[:, 2, 0::2, 0::2] = conv_cfa[:, 3, 0::2, 0::2]
        else:
            output = torch.cat(
                (
                    x[:, 0:1, 0::2, 0::2],
                    (x[:, 0:1, 0::2, 1::2] + x[:, 0:1, 1::2, 0::2]) / 2,
                    x[:, 0:1, 1::2, 1::2],
                ),
                1,
            )  # 1x3xW/2xH/2
        return output.clamp_(0, 1)

    def reverse(self, x):
        if self.NeedDemosaic:
            output = torch.zeros_like(x[:, 1:2, :, :])
            output[..., 0::2, 0::2] = x[:, 0, 0::2, 0::2]
            output[..., 0::2, 1::2] = x[:, 1, 0::2, 1::2]
            output[..., 1::2, 0::2] = x[:, 1, 1::2, 0::2]
            output[..., 1::2, 1::2] = x[:, 2, 1::2, 1::2]
        else:
            output = torch.zeros(
                x.size(0), x.size(1) // 3, x.size(2) * 2, x.size(3) * 2
            )
            output[..., 0::2, 0::2] = x[:, 0, ...]
            output[..., 0::2, 1::2] = x[:, 1, ...]
            output[..., 1::2, 0::2] = x[:, 1, ...]
            output[..., 1::2, 1::2] = x[:, 2, ...]
        return output.clamp_(0, 1)


class EnhanceSaturation(nn.Module):
    """
    RGB --> HSV --> RGB
    """

    def __init__(self, SaturationFactor=1):
        super(EnhanceSaturation, self).__init__()
        self.saturationfactor = SaturationFactor

    def forward(self, x):
        hsv = utils_color.rgb2hsv(x.clamp_(0, 1), True)
        #        hsv[:,0,...] -= 10
        #        hsv[:,0,...] %= 360

        hsv[:, 1, ...] *= self.saturationfactor
        hsv[:, 1, ...].clamp_(0, 1)
        output = utils_color.rgb2hsv(hsv, False)
        return output.clamp_(0, 1)

    def reverse(self, x):
        hsv = utils_color.rgb2hsv(x.clamp_(0, 1), True)
        #        hsv[:,0,...] += 10
        #        hsv[:,0,...] %= 360
        hsv[:, 1, ...] /= self.saturationfactor
        output = utils_color.rgb2hsv(hsv, False)
        return output.clamp_(0, 1)


class ToneMapping(nn.Module):
    """
    Tone Mapping
    """

    def __init__(self, ToneCurveX, ToneCurveY, delta=1e-6):
        super(ToneMapping, self).__init__()
        self.delta = delta
        xi = np.linspace(0, 1, num=int(1 / delta + 1), endpoint=True)
        yi = interp1d(ToneCurveX, ToneCurveY, kind="cubic")(xi)
        yi_inv = interp1d(yi, xi, kind="cubic")(xi)
        self.yi = torch.from_numpy(yi).float()
        self.yi_inv = torch.from_numpy(yi_inv).float()

    def forward(self, x):
        x = self.yi[(torch.round(x.clamp_(0, 1) / self.delta)).long()]
        return x.clamp_(0, 1)

    def reverse(self, x):
        x = self.yi_inv[(torch.round(x.clamp_(0, 1) / self.delta)).long()]
        return x.clamp_(0, 1)


class AddNoise(nn.Module):
    """
    Tone Mapping
    """

    def __init__(self):
        super(AddNoise, self).__init__()

    def forward(self, x):
        return x.clamp_(0, 1)

    def reverse(self, x):
        shot_noise, read_noise = utils_noise.random_noise_levels_dnd()
        x = utils_noise.add_noise(x, shot_noise, read_noise, use_cuda=False)
        return x.clamp_(0, 1)


class ToneMapping2(nn.Module):
    """
    Tone Mapping
    """

    def __init__(self, ToneCurveX, ToneCurveY, delta=1e-6):
        super(ToneMapping2, self).__init__()
        self.delta = delta
        xi = np.linspace(0, 1, num=int(1 / delta + 1), endpoint=True)
        yi = interp1d(ToneCurveX, ToneCurveY, kind="quadratic")(xi)
        yi_inv = interp1d(yi, xi, kind="quadratic")(xi)
        self.yi = torch.from_numpy(yi).float()
        self.yi_inv = torch.from_numpy(yi_inv).float()

    def forward(self, x):
        srcLum = torch.mean(x.clamp_(0, 1), 1, keepdim=True)
        dstLum = self.yi[(torch.round(srcLum / self.delta)).long()]
        x = x * (dstLum / srcLum)
        return x.clamp_(0, 1)

    def reverse(self, x):
        x1 = torch.mean(x.clamp_(0, 1), 1, keepdim=True)
        x2 = self.yi_inv[(torch.round(x1.clamp_(0, 1) / self.delta)).long()]
        x = x / (x1 / x2)
        return x.clamp_(0, 1)


class ISPNet(nn.Module):
    def __init__(
        self,
        weight_raw2xyz,
        ToneCurveX,
        ToneCurveY,
        BaselineExposure=0,
        BaselineExposureOffset=0,
        ColorSpace=0,
        NeedDemosaic=True,
    ):
        super(ISPNet, self).__init__()

        self.exposurecompensation = ExposureCompensation(
            BaselineExposure=BaselineExposure,
            BaselineExposureOffset=BaselineExposureOffset,
        )
        self.raw2xyz = Raw2XYZ(weight=weight_raw2xyz)
        self.xyz2linearrgb = XYZ2LinearRGB(ColorSpace=ColorSpace)
        self.tonemapping = ToneMapping(ToneCurveX=ToneCurveX, ToneCurveY=ToneCurveY)
        self.gammacorrect = GammaCorrect(ColorSpace=ColorSpace)
        self.demosaic = Demosaic(NeedDemosaic=NeedDemosaic)
        self.addnoise = AddNoise()

    def forward(self, x, for_noisy=True):
        if for_noisy:
            x = self.demosaic.forward(
                x
            )  # 1) demosaicing, camera space # may lead to color shift for some images
        x = self.exposurecompensation.forward(
            x
        )  # 2) exposure compensation, camera space
        x = self.raw2xyz.forward(x)  # 3) camera2xyz(d50), XYZ space
        x = self.xyz2linearrgb.forward(x)  # 7) xyz(d50) --> target linear RGB
        x = self.tonemapping.forward(x)  # 8) tone mapping, target linear rgb space
        x = self.gammacorrect.forward(x)  # 9) gamma correction, target rgb space
        return x

    def reverse(self, x, for_noisy=True):
        x = self.gammacorrect.reverse(x)
        x = self.tonemapping.reverse(x)
        x = self.xyz2linearrgb.reverse(x)
        x = self.raw2xyz.reverse(x)
        x = self.exposurecompensation.reverse(x)
        if for_noisy:
            x = self.demosaic.reverse(x)
            x = self.addnoise.reverse(x)
        return x


class ISPModel(nn.Module):
    def __init__(self):
        super(ISPModel, self).__init__()
        self.ToneCurves = mat.loadmat(
            os.path.join(CAMERA_PROFILE_DIR, "tonecurves.mat")
        )[
            "ToneCurves"
        ]  # 203 tone curves
        self.CameraType = [
            "canon_eos_1d_mark_ii",
            "canon_eos_5d_mark_iii",
            "canon",
            "canon_eos_6d_v1",
            "huawei_p20",
            "huawei_p30",
            "huawei_v8",
            "nikon_d500",
            "nikon_d810",
            "nikon_d5600",
            "olympus_em1",
        ]
        self.count = 0

    def forward(self, x, x1):
        x = util.single2tensor4(x)
        x1 = util.single2tensor4(x1)
        if self.count % 64 == 0:
            CameraType = random.choice(self.CameraType)
            ProfileType = CameraType
            self.CameraProfile = mat.loadmat(
                os.path.join(CAMERA_PROFILE_DIR, ProfileType + ".mat")
            )

            #            if ProfileType == 'huawei_v8':
            #                ToneCurve_idx = 74
            #            elif ProfileType == 'huawei_p20':
            #                ToneCurve_idx = 127 # 115 127 128 132 133
            #            elif ProfileType == 'huawei_p30':
            #                ToneCurve_idx = 126  # 66, 126
            #            elif 'nikon' in ProfileType:
            #                ToneCurve_idx = 1
            #            elif 'canon' in ProfileType:
            #                ToneCurve_idx = 2
            #            elif 'olympus' in ProfileType:
            #                ToneCurve_idx = 0
            ToneCurve_idx = np.random.choice(
                [0, 1, 2, 66, 126, 115, 127, 128, 132, 133, 74]
            )
            ToneCurve = self.ToneCurves[ToneCurve_idx, :]
            ToneCurve = np.reshape(ToneCurve, (2, -1), "F")
            ToneCurveX, ToneCurveY = ToneCurve[0, :], ToneCurve[1, :]
            self.ForwardMatrix1 = (
                torch.from_numpy(self.CameraProfile["ForwardMatrix1"])
                .float()
                .reshape(3, 3)
            )
            self.ForwardMatrix2 = (
                torch.from_numpy(self.CameraProfile["ForwardMatrix2"])
                .float()
                .reshape(3, 3)
            )
            FW = torch.rand(1)
            D = (
                torch.tensor(
                    [1.2 + 1.2 * torch.rand(1), 1.0, 1.2 + 1.2 * torch.rand(1)]
                )
                .diag()
                .float()
            )
            CameraToXYZ_D50 = torch.matmul(
                FW * self.ForwardMatrix1 + (1 - FW) * self.ForwardMatrix2, D
            )
            BaselineExposureOffset = 0.2 * torch.rand(1) - 0.1
            self.isp = ISPNet(
                weight_raw2xyz=CameraToXYZ_D50,
                ToneCurveX=ToneCurveX,
                ToneCurveY=ToneCurveY,
                BaselineExposure=0,
                BaselineExposureOffset=BaselineExposureOffset,
            )
        self.count += 1

        img_raw = self.isp.reverse(x, True)
        x = self.isp.forward(img_raw, True)

        img_raw = self.isp.reverse(x1, False)
        x1 = self.isp.forward(img_raw, False)

        x = util.tensor2single(x)
        x1 = util.tensor2single(x1)
        return x, x1


if __name__ == "__main__":

    img = util.imread_uint("utils/test.bmp", 3)
    util.imsave(img, "a0.png")
    img = util.uint2single(img)

    ToneCurves = mat.loadmat(os.path.join(CAMERA_PROFILE_DIR, "tonecurves.mat"))[
        "ToneCurves"
    ]  # 203 tone curves
    CameraType = [
        "canon_eos_1d_mark_ii",
        "canon_eos_5d_mark_iii",
        "canon",
        "canon_eos_6d_v1",
        "huawei_p20",
        "huawei_p30",
        "huawei_v8",
        "nikon_d500",
        "nikon_d810",
        "nikon_d5600",
        "olympus_em1",
    ]
    CameraType = random.choice(CameraType)
    ProfileType = CameraType
    CameraProfile = mat.loadmat(os.path.join(CAMERA_PROFILE_DIR, ProfileType + ".mat"))

    if ProfileType == "huawei_v8":
        ToneCurve_idx = 74
    elif ProfileType == "huawei_p20":
        ToneCurve_idx = 127  # 115 127 128 132 133
    elif ProfileType == "huawei_p30":
        ToneCurve_idx = 126  # 66, 126
    elif "nikon" in ProfileType:
        ToneCurve_idx = 1
    elif "canon" in ProfileType:
        ToneCurve_idx = 2
    elif "olympus" in ProfileType:
        ToneCurve_idx = 0
    ToneCurve = ToneCurves[ToneCurve_idx, :]
    ToneCurve = np.reshape(ToneCurve, (2, -1), "F")
    ToneCurveX, ToneCurveY = ToneCurve[0, :], ToneCurve[1, :]
    ForwardMatrix1 = (
        torch.from_numpy(CameraProfile["ForwardMatrix1"]).float().reshape(3, 3)
    )
    ForwardMatrix2 = (
        torch.from_numpy(CameraProfile["ForwardMatrix2"]).float().reshape(3, 3)
    )
    FW = torch.rand(1)
    D = (
        torch.tensor([1.2 + 2 * torch.rand(1), 1.0, 1.2 + 2 * torch.rand(1)])
        .diag()
        .float()
    )
    CameraToXYZ_D50 = torch.matmul(FW * ForwardMatrix1 + (1 - FW) * ForwardMatrix2, D)
    BaselineExposureOffset = 0.4 * torch.rand(1) - 0.2
    isp = ISPNet(
        weight_raw2xyz=CameraToXYZ_D50,
        ToneCurveX=ToneCurveX,
        ToneCurveY=ToneCurveY,
        BaselineExposure=0,
        BaselineExposureOffset=BaselineExposureOffset,
    )

    img = util.single2tensor4(img)

    img_raw = isp.reverse(img.clone())
    img = isp.forward(img_raw)

    img = util.tensor2uint(img)
    util.imsave(img, "a.png")
    # run utils/utils_isp.py


#    ColorSpace = 0                       # 0 for sRGB color space, 1 for Adobe RGB, 2 for Pro Photo RGB
#    NeedDemosaic = False                 # matlab function 'demosaic', set False to get half image
#    ForwardMatrixWeightFactor = 1        # from [0, 1], slight difference between setting 0 and 1
#    BaselineExposureOffset = 0           # adjust exposure
#
#    if ProfileType == 'huawei_v8':
#        ToneCurve_idx = 74
#    elif ProfileType == 'huawei_p20':
#        ToneCurve_idx = 127 # 115 127 128 132 133
#    elif ProfileType == 'huawei_p30':
#        ToneCurve_idx = 126  # 66, 126
#    elif 'nikon' in ProfileType:
#        ToneCurve_idx = 1
#    elif 'canon' in ProfileType:
#        ToneCurve_idx = 2
#    elif 'olympus' in ProfileType:
#        ToneCurve_idx = 0
#
#    #ToneCurve_idx = 66                # 1, 2,  3, 74, 66, 126, 115 127 128 132 133
#    ToneCurve = ToneCurves[ToneCurve_idx, :]
#    ToneCurve = np.reshape(ToneCurve, (2, -1), 'F')
#    ToneCurveX, ToneCurveY = ToneCurve[0, :], ToneCurve[1, :]
#
#    ForwardMatrix1 = torch.from_numpy(CameraProfile['ForwardMatrix1']).float().reshape(3,3)
#    ForwardMatrix2 = torch.from_numpy(CameraProfile['ForwardMatrix2']).float().reshape(3,3)
#
#    D = torch.inverse(ReferenceNeutral.squeeze().diag())
#
#    CameraToXYZ_D50 = torch.matmul(ForwardMatrixWeightFactor*ForwardMatrix1 + (1-ForwardMatrixWeightFactor)*ForwardMatrix2, D)
#
#
#    isp = ISPNet(weight_raw2xyz=CameraToXYZ_D50,
#                 ToneCurveX=ToneCurveX, ToneCurveY=ToneCurveY,
#                 BaselineExposure=0, BaselineExposureOffset=0,
#                 ColorSpace=0,
#                 NeedDemosaic=True)
#
#
#    x, y = isp.reverse(x)
