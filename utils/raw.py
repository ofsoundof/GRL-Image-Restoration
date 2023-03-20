#!/usr/bin/env python3
import numpy as np
import torch


BAYER_PATTERNS = ["RGGB", "GRBG", "GBRG", "BGGR"]
NORMALIZATION_MODE = ["crop", "pad"]

(BAYER_R, BAYER_GR, BAYER_GB, BAYER_B) = range(4)
BAYER_INIT_XY = ((0, 0), (1, 0), (0, 1), (1, 1))


def pack_raw_batch(raw_batch):
    upscale_factor = 2
    shape_prefix = raw_batch.shape[0:-3]
    channels, in_height, in_width = raw_batch.shape[-3:]
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = raw_batch.contiguous().view(
        *shape_prefix, channels, out_height, upscale_factor, out_width, upscale_factor
    )

    channels *= upscale_factor**2
    prefix_len = len(shape_prefix)
    permute_list = list(range(prefix_len)) + [
        0 + prefix_len,
        2 + prefix_len,
        4 + prefix_len,
        1 + prefix_len,
        3 + prefix_len,
    ]
    unshuffle_out = input_view.permute(permute_list).contiguous()
    return unshuffle_out.view(*shape_prefix, channels, out_height, out_width)


def unpack_raw(im):
    bs, chan, h, w = im.shape
    H, W = h * 2, w * 2
    # Init tensors using type_as
    img2 = torch.zeros((bs, H, W))
    img2 = img2.type_as(im)
    img2[:, 0:H:2, 0:W:2] = im[:, 0, :, :]
    img2[:, 0:H:2, 1:W:2] = im[:, 1, :, :]
    img2[:, 1:H:2, 0:W:2] = im[:, 2, :, :]
    img2[:, 1:H:2, 1:W:2] = im[:, 3, :, :]
    img2 = img2.unsqueeze(1)
    return img2


def demosaic_bilinear(in_frame, bayer_order):
    """Bilinear interpolate bayer raw to RGB.

    Args:
        in_frame: Bayer format numpy(h,w) u8, u16 image.

    Returns:
        out_frame: RGB numpy frame (h, w, 3), u8, u16 image.
    """

    h, w = in_frame.shape[:2]
    hh, hw = h >> 1, w >> 1
    out_frame = np.empty((hh, 2, hw, 2, 3), dtype=np.uint16)
    bayer_frame = in_frame.reshape(hh, 2, hw, 2)
    # R, B channel
    for raw_ch, rgb_ch in zip([BAYER_R, BAYER_B], [0, 2]):
        bayer = bayer_order ^ raw_ch
        x, y = bayer & 1, (bayer >> 1) & 1
        ch_rgb = out_frame[:, :, :, :, rgb_ch]
        ch_rgb[:, y, :, x] = bayer_frame[:, y, :, x]
        if bool(x):
            ch_rgb[:, y, 1:, 0] = (ch_rgb[:, y, :-1, 1] + ch_rgb[:, y, 1:, 1]) // 2
            ch_rgb[:, y, 0, 0] = ch_rgb[:, y, 0, 1]
        else:
            ch_rgb[:, y, :-1, 1] = (ch_rgb[:, y, :-1, 0] + ch_rgb[:, y, 1:, 0]) // 2
            ch_rgb[:, y, -1, 1] = ch_rgb[:, y, -1, 0]
        if bool(y):
            ch_rgb[1:, 0, :, :] = (ch_rgb[:-1, 1, :, :] + ch_rgb[1:, 1, :, :]) // 2
            ch_rgb[0, 0, :, :] = ch_rgb[0, 1, :, :]
        else:
            ch_rgb[:-1, 1, :, :] = (ch_rgb[:-1, 0, :, :] + ch_rgb[1:, 0, :, :]) // 2
            ch_rgb[-1, 1, :, :] = ch_rgb[-1, 0, :, :]
    # G channel
    for raw_ch in [BAYER_GR, BAYER_GB]:
        bayer = bayer_order ^ raw_ch
        x, y = bayer & 1, (bayer >> 1) & 1
        out_frame[:, y, :, x, 1] = bayer_frame[:, y, :, x]

    for raw_ch in [BAYER_GR, BAYER_GB]:
        bayer = bayer_order ^ raw_ch
        x, y = bayer & 1, (bayer >> 1) & 1
        ch_rgb = out_frame[:, :, :, :, 1]
        rx = x ^ 1
        ry = y ^ 1
        if x == 0 and y == 0:
            ch_rgb[1:, y, :-1, rx] = (
                ch_rgb[1:, y, :-1, x]
                + ch_rgb[1:, y, 1:, x]
                + ch_rgb[:-1, ry, :-1, rx]
                + ch_rgb[1:, ry, :-1, rx]
            ) // 4
            ch_rgb[0, y, :-1, rx] = (ch_rgb[0, y, :-1, x] + ch_rgb[0, y, 1:, x]) // 2
            ch_rgb[1:, y, -1, rx] = (
                ch_rgb[:-1, ry, -1, rx] + ch_rgb[1:, ry, -1, rx]
            ) // 2
            ch_rgb[-1, y, -1, rx] = (ch_rgb[-1, y, -1, x] + ch_rgb[-1, ry, -1, rx]) // 2
        if x == 1 and y == 0:
            ch_rgb[1:, y, 1:, rx] = (
                ch_rgb[1:, y, :-1, x]
                + ch_rgb[1:, y, 1:, x]
                + ch_rgb[:-1, ry, 1:, rx]
                + ch_rgb[1:, ry, 1:, rx]
            ) // 4
            ch_rgb[0, y, 1:, rx] = (ch_rgb[0, y, :-1, x] + ch_rgb[0, y, 1:, x]) // 2
            ch_rgb[1:, y, 0, rx] = (ch_rgb[:-1, ry, 0, rx] + ch_rgb[1:, ry, 0, rx]) // 2
            ch_rgb[0, y, 0, rx] = (ch_rgb[0, y, 0, x] + ch_rgb[0, ry, 0, rx]) // 2
        if x == 0 and y == 1:
            ch_rgb[:-1, y, :-1, rx] = (
                ch_rgb[:-1, y, :-1, x]
                + ch_rgb[:-1, y, 1:, x]
                + ch_rgb[:-1, ry, :-1, rx]
                + ch_rgb[1:, ry, :-1, rx]
            ) // 4
            ch_rgb[-1, y, :-1, rx] = (ch_rgb[-1, y, :-1, x] + ch_rgb[-1, y, 1:, x]) // 2
            ch_rgb[:-1, y, -1, rx] = (
                ch_rgb[:-1, ry, -1, rx] + ch_rgb[1:, ry, -1, rx]
            ) // 2
            ch_rgb[-1, y, -1, rx] = (ch_rgb[-1, y, -1, x] + ch_rgb[-1, ry, -1, rx]) // 2
        if x == 1 and y == 1:
            ch_rgb[:-1, y, 1:, rx] = (
                ch_rgb[:-1, y, :-1, x]
                + ch_rgb[:-1, y, 1:, x]
                + ch_rgb[:-1, ry, 1:, rx]
                + ch_rgb[1:, ry, 1:, rx]
            ) // 4
            ch_rgb[:-1, y, 0, rx] = (
                ch_rgb[:-1, ry, 0, rx] + ch_rgb[1:, ry, 0, rx]
            ) // 2
            ch_rgb[-1, y, 1:, rx] = (ch_rgb[-1, y, :-1, x] + ch_rgb[-1, y, 1:, x]) // 2
            ch_rgb[-1, y, 0, rx] = (ch_rgb[-1, y, 0, x] + ch_rgb[-1, ry, 0, rx]) // 2

    return out_frame.reshape(h, w, 3)


def apply_digital_gain(im):
    arbitrary_brightness = 0.15
    mean = torch.mean(im, list(range(im.ndim))[1:])
    multipliers = arbitrary_brightness / mean
    multipliers.clip(1.0)  # don't darken
    im *= multipliers.view(
        multipliers.size(0), *[1 for i in range(im.ndim - 1)]
    )  # broadcast multipliers
    return im.clip(0.0, 1.0)


def bayer_unify(
    raw: np.ndarray, input_pattern: str, target_pattern: str, mode: str
) -> np.ndarray:
    """
    Convert a bayer raw image from one bayer pattern to another.
    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be unified.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    target_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The expected output pattern.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """
    if input_pattern not in BAYER_PATTERNS:
        raise ValueError("Unknown input bayer pattern!")
    if target_pattern not in BAYER_PATTERNS:
        raise ValueError("Unknown target bayer pattern!")
    if mode not in NORMALIZATION_MODE:
        raise ValueError("Unknown normalization mode!")
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError("raw should be a 2-dimensional numpy.ndarray!")

    if input_pattern == target_pattern:
        h_offset, w_offset = 0, 0
    elif (
        input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]
    ):
        h_offset, w_offset = 1, 0
    elif (
        input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]
    ):
        h_offset, w_offset = 0, 1
    elif (
        input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]
    ):
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError("Unexpected pair of input and target bayer pattern!")

    if mode == "pad":
        out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], "reflect")
    elif mode == "crop":
        h, w = raw.shape
        out = raw[h_offset : h - h_offset, w_offset : w - w_offset]
    else:
        raise ValueError("Unknown normalization mode!")

    return out


def bayer_aug(
    raw: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool, input_pattern: str
) -> np.ndarray:
    """
    Apply augmentation to a bayer raw image.
    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be augmented. H and W must be even numbers.
    flip_h : bool
        If True, do vertical flip.
    flip_w : bool
        If True, do horizontal flip.
    transpose : bool
        If True, do transpose.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    """

    if input_pattern not in BAYER_PATTERNS:
        raise ValueError("Unknown input bayer pattern!")
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError("raw should be a 2-dimensional numpy.ndarray")
    if raw.shape[0] % 2 == 1 or raw.shape[1] % 2 == 1:
        raise ValueError("raw should have even number of height and width!")

    aug_pattern, target_pattern = input_pattern, input_pattern

    out = raw
    if flip_h:
        out = out[::-1, :]
        aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
    if flip_w:
        out = out[:, ::-1]
        aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
    if transpose:
        out = out.T
        aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]

    out = bayer_unify(out, aug_pattern, target_pattern, "crop")
    return out
