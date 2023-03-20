import torch

"""
CIELab(D50) is more perceptually uniform than CIEXYZ(D50).
Unlike the RGB and CMYK color models, Lab color is designed to approximate human vision. It aspires to perceptual uniformity, and its L component closely matches human perception of lightness, although it does not take the Helmholtz–Kohlrausch effect into account. Thus, it can be used to make accurate color balance corrections by modifying output curves in the a and b components, or to adjust the lightness contrast using the L component.
http://www.brucelindbloom.com/
https://en.wikipedia.org/wiki/CIELAB_color_space
https://en.wikipedia.org/wiki/CIE_1931_color_space
sRGB:            | The default space of Windows and the Internet. Limited color gamut based on typical CRT phosphors. Gamma = 2.2 (approximately), White point = 6500K (D65).
Adobe RGB (1998) | Medium gamut, with stronger greens than sRGB. Often recommended for high quality printed output. Gamma = 2.2, White point = 6500K (D65).
Wide Gamut RGB	 | Extremely wide gamut with primaries on the spectral locus at 450, 525, and 700 microns. One of the color spaces supported by the Canon DPP RAW converter. 48-bit color files are recommended with wide gamut spaces: banding can be a problem with 24-bit color. Gamma = 2.2, White point = 5000K (D50).
ProPhoto RGB	 | Extremely wide gamut. Gamma = 1.8, White point = 5000K (D50). Described in RIMM/ROMM RGB Color Encodings by Spaulding, Woolfe and Giorgianni.
Apple RGB	     | Small gamut. Used by Apple. Gamma = 1.8, White point = 6500K (D65).
ColorMatch RGB	 | Small gamut. Used by Apple. Gamma = 1.8, White point = 5000K (D50).
Rec. 709 Legal	 | Small gamut. Used in HDTV. Pixel values 16-235.
Rec. 709 Full	 | Same as Rec. 709 Legal, but with Pixel values 0-255.
ACES	         | Academy Color Encoding System, used in the workflow developed by the folks who bring you the Oscars. Extremely large gamut, covering all visible colors. Linear gamma. White point = 6000K.
Rec. 2020 Legal	 | Fairly large gamut, covering most colors from reflected objects. For UHDTV. Pixel values 16-235.
Rec. 2020 Full	 | Same as Rec. 2020 Legal, but with Pixel values 0-255.
DCI-P3	         | Digital cinema projection color space. 25% wider gamut than sRGB, covering most reflective surface colors. Gamma = 2.6.
Display p3	     | Used in the iPhone 8. Same gamut as DCI-P3, but gamma is approximately 2.2 (same as sRGB). May be a part of the new Apple HEIF file format, intended to replace JPEG. This is new information (as of late 2017). Reliable information is hard to come by.
"""


def xyz2linearrgb_weight(colorspace=0, is_forward=True):
    # reference: http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    # color space: sRGB(D65) < Adobe RGB(D65) < ProPhoto RGB(D50)
    # XYZ space: XYZ(D50)
    D50 = torch.FloatTensor([0.96422, 1, 0.82521]).reshape(3, 1)
    D65 = torch.FloatTensor([0.95047, 1, 1.08883]).reshape(3, 1)
    if colorspace == "sRGB" or colorspace == 0:
        Madapt = MatBradfordAlgorithm(D50, D65)
        xyz2rgb_sRGB = torch.FloatTensor(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )
        xyz2rgb = torch.matmul(xyz2rgb_sRGB, Madapt)
    elif colorspace == "Adobe RGB" or colorspace == 1:
        Madapt = MatBradfordAlgorithm(D50, D65)
        xyz2rgb_Adobe_RGB = torch.FloatTensor(
            [
                [2.0413690, -0.5649464, -0.3446944],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0134474, -0.1183897, 1.0154096],
            ]
        )
        xyz2rgb = torch.matmul(xyz2rgb_Adobe_RGB, Madapt)
    elif colorspace == "ProPhoto RGB" or colorspace == 2:
        xyz2rgb = torch.FloatTensor(
            [
                [1.3459433, -0.2556075, -0.0511118],
                [-0.5445989, 1.5081673, 0.0205351],
                [0.0, 0.0, 1.2118128],
            ]
        )

    else:
        Madapt = MatBradfordAlgorithm(D50, D65)
        xyz2rgb_sRGB = torch.FloatTensor(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        )
        xyz2rgb = torch.matmul(xyz2rgb_sRGB, Madapt)

    if not is_forward:
        xyz2rgb = torch.inverse(xyz2rgb)

    return xyz2rgb


def MatBradfordAlgorithm(XYZ, XYZw, method="Bradford"):
    # Chromatic adaptation between different reference white
    # http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    if method == "Bradford" or method == 0:
        Mbfd = torch.FloatTensor(
            [
                [0.8951, 0.2664, -0.1614],
                [-0.7502, 1.7135, 0.0367],
                [0.0389, -0.0685, 1.0296],
            ]
        )
    elif method == "XYZ Scaling" or method == 1:
        Mbfd = torch.eye(3).float()
    elif method == "Von Kries" or method == 2:
        Mbfd = torch.FloatTensor(
            [
                [0.40024, 0.7076, -0.08081],
                [-0.2263, 1.16532, 0.0457],
                [0.0, 0.0, 0.91822],
            ]
        )
    elif method == 3:
        Mbfd = torch.FloatTensor(
            [
                [0.8562, 0.3372, -0.1934],
                [-0.8360, 1.8327, 0.0033],
                [0.0357, -0.0469, 1.0112],
            ]
        )
    else:
        Mbfd = torch.FloatTensor(
            [
                [0.7328, 0.4296, -0.1624],
                [-0.7036, 1.6975, 0.0061],
                [0.0030, 0.0136, 0.9834],
            ]
        )
    SRC = torch.matmul(Mbfd, XYZ)
    PCS = torch.matmul(Mbfd, XYZw)
    T = (PCS / SRC).diagflat()
    Madapt = torch.matmul(torch.matmul(torch.inverse(Mbfd), T), Mbfd)
    return Madapt


def rgb2hsv(img, is_forward=True):
    """
    Conversion between RGB ([0, 1]) and HSV (h([0,360]), s([0,1]), v([0,1]))
    img: NxCxWxH
    out: NxCxWxH

    Example:
        img = torch.rand(2,3,4000,5000).mul_(255.).int().div_(255.).float()
        out = rgb2hsv(rgb2hsv(img, True), False)
        error_sum = torch.sum(torch.abs(img-out))
        print(error_sum)
    """
    if is_forward:
        maxv, maxd = torch.max(img, dim=1, keepdim=False)
        minv, mind = torch.min(img, dim=1, keepdim=False)
        out = torch.zeros_like(img)
        out[:, 0, ...][maxd == mind] = torch.zeros_like(out[:, 0, ...][maxd == mind])
        out[:, 0, ...][maxd == 0] = (
            ((img[:, 1, ...] - img[:, 2, ...]) * 60.0 / (maxv - minv + 1e-8)) % 360.0
        )[maxd == 0]
        out[:, 0, ...][maxd == 1] = (
            ((img[:, 2, ...] - img[:, 0, ...]) * 60.0 / (maxv - minv + 1e-8)) + 120.0
        )[maxd == 1]
        out[:, 0, ...][maxd == 2] = (
            ((img[:, 0, ...] - img[:, 1, ...]) * 60.0 / (maxv - minv + 1e-8)) + 240.0
        )[maxd == 2]
        out[:, 1, ...][maxv == 0] = torch.zeros_like(out[:, 1, ...][maxv == 0])
        out[:, 1, ...][maxv != 0] = (1 - minv / maxv)[maxv != 0]
        out[:, 2, ...] = maxv
    else:
        img = img.permute(1, 0, 2, 3)
        hi = torch.floor(img[0, ...] / 60.0) % 6
        hi = hi.int()
        v = img[2, ...]
        f = (img[0, ...] / 60.0) - torch.floor(img[0, ...] / 60.0)
        p = v * (1.0 - img[1, ...])
        q = v * (1.0 - (f * img[1, ...]))
        t = v * (1.0 - ((1.0 - f) * img[1, ...]))
        out = torch.zeros_like(img)
        out[..., hi == 0] = torch.stack((v, t, p), 0)[..., hi == 0]
        out[..., hi == 1] = torch.stack((q, v, p), 0)[..., hi == 1]
        out[..., hi == 2] = torch.stack((p, v, t), 0)[..., hi == 2]
        out[..., hi == 3] = torch.stack((p, q, v), 0)[..., hi == 3]
        out[..., hi == 4] = torch.stack((t, p, v), 0)[..., hi == 4]
        out[..., hi == 5] = torch.stack((v, p, q), 0)[..., hi == 5]
        out = out.permute(1, 0, 2, 3)
    return out


def linear2gamma(x, colorspace=0, is_forward=True):

    if colorspace == "sRGB" or colorspace == 0:
        if is_forward:
            idx = x > 0.0031308
            x[idx] = 1.055 * torch.pow(x[idx], 1.0 / 2.4) - 0.055
            x[~idx] = 12.92 * x[~idx]
        else:
            idx = x > 0.04045
            x[~idx] = x[~idx].clamp(min=1e-8) / 12.92
            x[idx] = torch.pow((200.0 * x[idx] + 11.0) / 211.0, 2.4)
    elif colorspace == "Adobe RGB" or colorspace == 1:
        r = 2.19921875
        if is_forward:
            x = torch.pow(x, 1 / r)
        else:
            x = torch.pow(x, r)
    elif colorspace == "ProPhoto RGB" or colorspace == 2:
        if is_forward:
            k = 16 ** (1.8 / (1 - 1.8))
            idx = x >= k
            x[idx] = torch.pow(x[idx], 1.0 / 1.8)
            x[~idx] = 16.0 * x[~idx]
        else:
            k = 16 ** (1.8 / (1 - 1.8)) * 16
            idx = x >= k
            x[idx] = torch.pow(x[idx], 1.8)
            x[~idx] = x[~idx] / 16.0
    else:
        if is_forward:
            idx = x > 0.0031308
            x[idx] = 1.055 * torch.pow(x[idx], 1.0 / 2.4) - 0.055
            x[~idx] = 12.92 * x[~idx]
        else:
            idx = x > 0.04045
            x[~idx] = x[~idx].clamp(min=1e-8) / 12.92
            x[idx] = torch.pow((200.0 * x[idx] + 11.0) / 211.0, 2.4)
    return x


def HsvLookup(img, HsvCalibration, HSVDivs):
    """
    img: NxCxWxH, in hsv space
    out: NxCxWxH
    """
    hd, sd, vd = HSVDivs  # [90, 30, 1]
    img = img.permute(1, 0, 2, 3)

    h = img[0, ...] / 360 * (hd - 1)
    h0 = torch.floor(h).clamp(max=hd - 1)
    h1 = (h0 + 1) % (hd - 1)
    he = h - h0

    s = img[1, ...] * (sd - 1)
    s0 = torch.floor(s).clamp(max=sd - 2)
    s1 = s0 + 1
    se = s - s0

    if vd == 1:
        e0 = (
            HsvCalibration[:, (sd * h0 + s0).long()] * (1 - he)
            + HsvCalibration[:, (sd * h1 + s0).long()] * he
        )
        e1 = (
            HsvCalibration[:, (sd * h0 + s1).long()] * (1 - he)
            + HsvCalibration[:, (sd * h1 + s1).long()] * he
        )
        e = e0 * (1 - se) + e1 * se
    else:
        v = img[2, ...] * (vd - 1)
        v0 = torch.floor(v).clamp(max=vd - 2)
        v1 = v0 + 1
        ve = v - v0
        hsd = hd * sd
        e00 = (
            HsvCalibration[:, (hsd * v0 + sd * h0 + s0).long()] * (1 - ve)
            + HsvCalibration[:, (hsd * v1 + sd * h0 + s0).long()] * ve
        )
        e01 = (
            HsvCalibration[:, (hsd * v0 + sd * h0 + s1).long()] * (1 - ve)
            + HsvCalibration[:, (hsd * v1 + sd * h0 + s1).long()] * ve
        )
        e10 = (
            HsvCalibration[:, (hsd * v0 + sd * h1 + s0).long()] * (1 - ve)
            + HsvCalibration[:, (hsd * v1 + sd * h1 + s0).long()] * ve
        )
        e11 = (
            HsvCalibration[:, (hsd * v0 + sd * h1 + s1).long()] * (1 - ve)
            + HsvCalibration[:, (hsd * v1 + sd * h1 + s1).long()] * ve
        )
        e = (e00 * (1 - he) + e10 * he) * (1 - se) + (e01 * (1 - he) + e11 * he) * se

    img[0, ...].add_(e[0, ...]).remainder_(360)
    img[1:, ...].mul_(e[1:, ...]).clamp_(min=0.0, max=1.0)
    img = img.permute(1, 0, 2, 3)
    return img


# if __name__ == '__main__':
