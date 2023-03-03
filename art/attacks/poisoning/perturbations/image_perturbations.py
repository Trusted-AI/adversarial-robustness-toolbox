# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Adversarial perturbations designed to work for images.
"""
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def add_single_bd(x: np.ndarray, distance: int = 2, pixel_value: int = 1) -> np.ndarray:
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for a single image
    or a batch of images.

    :param x: A single image or batch of images of shape NWHC, NHW, or HC. Pixels will be added to all channels.
    :param distance: Distance from bottom-right walls.
    :param pixel_value: Value used to replace the entries of the image matrix.
    :return: Backdoored image.
    """
    x = np.copy(x)
    shape = x.shape
    if len(shape) == 4:
        height, width = x.shape[1:3]
        x[:, height - distance, width - distance, :] = pixel_value
    elif len(shape) == 3:
        height, width = x.shape[1:]
        x[:, height - distance, width - distance] = pixel_value
    elif len(shape) == 2:
        height, width = x.shape
        x[height - distance, width - distance] = pixel_value
    else:
        raise ValueError(f"Invalid array shape: {shape}")
    return x


def add_pattern_bd(x: np.ndarray, distance: int = 2, pixel_value: int = 1) -> np.ndarray:
    """
    Augments a matrix by setting a checkerboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.

    :param x: A single image or batch of images of shape NWHC, NHW, or HC. Pixels will be added to all channels.
    :param distance: Distance from bottom-right walls.
    :param pixel_value: Value used to replace the entries of the image matrix.
    :return: Backdoored image.
    """
    x = np.copy(x)
    shape = x.shape
    if len(shape) == 4:
        height, width = x.shape[1:3]
        x[:, height - distance, width - distance, :] = pixel_value
        x[:, height - distance - 1, width - distance - 1, :] = pixel_value
        x[:, height - distance, width - distance - 2, :] = pixel_value
        x[:, height - distance - 2, width - distance, :] = pixel_value
    elif len(shape) == 3:
        height, width = x.shape[1:]
        x[:, height - distance, width - distance] = pixel_value
        x[:, height - distance - 1, width - distance - 1] = pixel_value
        x[:, height - distance, width - distance - 2] = pixel_value
        x[:, height - distance - 2, width - distance] = pixel_value
    elif len(shape) == 2:
        height, width = x.shape
        x[height - distance, width - distance] = pixel_value
        x[height - distance - 1, width - distance - 1] = pixel_value
        x[height - distance, width - distance - 2] = pixel_value
        x[height - distance - 2, width - distance] = pixel_value
    else:
        raise ValueError(f"Invalid array shape: {shape}")
    return x


def insert_image(
    x: np.ndarray,
    backdoor_path: str = "../utils/data/backdoors/alert.png",
    channels_first: bool = False,
    random: bool = True,
    x_shift: int = 0,
    y_shift: int = 0,
    size: Optional[Tuple[int, int]] = None,
    mode: str = "L",
    blend=0.8,
) -> np.ndarray:
    """
    Augments a matrix by setting a checkerboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.

    :param x: A single image or batch of images of shape NHWC, NCHW, or HWC. Input is in range [0,1].
    :param backdoor_path: The path to the image to insert as a trigger.
    :param channels_first: Whether the channels axis is in the first or last dimension
    :param random: Whether or not the image should be randomly placed somewhere on the image.
    :param x_shift: Number of pixels from the left to shift the trigger (when not using random placement).
    :param y_shift: Number of pixels from the right to shift the trigger (when not using random placement).
    :param size: The size the trigger image should be (height, width). Default `None` if no resizing necessary.
    :param mode: The mode the image should be read in. See PIL documentation
                 (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes).
    :param blend: The blending factor
    :return: Backdoored image.
    """
    n_dim = len(x.shape)
    if n_dim == 4:
        return np.array(
            [
                insert_image(single_img, backdoor_path, channels_first, random, x_shift, y_shift, size, mode, blend)
                for single_img in x
            ]
        )

    if n_dim != 3:
        raise ValueError(f"Invalid array shape {x.shape}")

    original_dtype = x.dtype
    data = np.copy(x)
    if channels_first:
        data = data.transpose([1, 2, 0])

    height, width, num_channels = data.shape

    no_color = num_channels == 1
    orig_img = Image.new("RGBA", (width, height), 0)  # height and width are swapped for PIL
    backdoored_img = Image.new("RGBA", (width, height), 0)  # height and width are swapped for PIL

    if no_color:
        backdoored_input = Image.fromarray((data * 255).astype("uint8").squeeze(axis=2), mode=mode)
    else:
        backdoored_input = Image.fromarray((data * 255).astype("uint8"), mode=mode)

    orig_img.paste(backdoored_input)

    trigger = Image.open(backdoor_path).convert("RGBA")
    if size:
        trigger = trigger.resize(size)

    backdoor_width, backdoor_height = trigger.size  # height and width are swapped for PIL

    if backdoor_width > width or backdoor_height > height:
        raise ValueError("Backdoor does not fit inside original image")

    if random:
        x_shift = np.random.randint(width - backdoor_width)
        y_shift = np.random.randint(height - backdoor_height)

    backdoored_img.paste(trigger, (x_shift, y_shift), mask=trigger)
    composite = Image.alpha_composite(orig_img, backdoored_img)
    backdoored_img = Image.blend(orig_img, composite, blend)

    backdoored_img = backdoored_img.convert(mode)

    res = np.array(backdoored_img) / 255.0

    if no_color:
        res = np.expand_dims(res, 2)

    if channels_first:
        res = res.transpose([2, 0, 1])

    return res.astype(original_dtype)
