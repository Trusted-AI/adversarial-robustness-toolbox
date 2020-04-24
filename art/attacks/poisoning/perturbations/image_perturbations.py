"""
Adversarial perturbations designed to work for images.
"""
import numpy as np
from typing import Optional, Tuple

from PIL import Image


def add_single_bd(x: np.ndarray, distance: int = 2, pixel_value: int = 1) -> np.ndarray:
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
    or a batch of images.

    :param x: N X W X H matrix or W X H matrix
    :param distance: Distance from bottom-right walls.
    :param pixel_value: Value used to replace the entries of the image matrix.
    :return: Backdoored image.
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
    else:
        raise ValueError("Invalid array shape: " + str(shape))
    return x


def add_pattern_bd(
    x: np.ndarray, distance: int = 2, pixel_value: int = 1
) -> np.ndarray:
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.

    :param x: N X W X H matrix or W X H matrix. will apply to last 2.
    :param distance: Distance from bottom-right walls.
    :param pixel_value: Value used to replace the entries of the image matrix.
    :return: Backdoored image.
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 3:
        width, height = x.shape[1:]
        x[:, width - distance, height - distance] = pixel_value
        x[:, width - distance - 1, height - distance - 1] = pixel_value
        x[:, width - distance, height - distance - 2] = pixel_value
        x[:, width - distance - 2, height - distance] = pixel_value
    elif len(shape) == 2:
        width, height = x.shape
        x[width - distance, height - distance] = pixel_value
        x[width - distance - 1, height - distance - 1] = pixel_value
        x[width - distance, height - distance - 2] = pixel_value
        x[width - distance - 2, height - distance] = pixel_value
    else:
        raise ValueError("Invalid array shape: " + str(shape))
    return x


def insert_image(
    x: np.ndarray,
    backdoor_path: str = "data/backdoors/post_it.png",
    random: bool = True,
    x_shift: int = 0,
    y_shift: int = 0,
    size: Optional[Tuple[int, int]] = None,
    mode: str = "L",
) -> np.ndarray:
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.

    :param x: N X W X H matrix or W X H matrix.
    :param backdoor_path: The path to the image to insert as a backdoor.
    :param random: Whether or not the image should be randomly placed somewhere on the image.
    :param x_shift: Number of pixels from the left to shift the backdoor (when not using random placement).
    :param y_shift: Number of pixels from the right to shift the backdoor (when not using random placement).
    :param size: The size the backdoor image should be (width, height). Default `None` if no resizing necessary.
    :param mode: The mode the image should be read in. See PIL documentation
                 (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes).
    :return: Backdoored image.
    """
    if len(x.shape) == 3:
        return np.array(
            [
                insert_image(single_img, backdoor_path, random, x_shift, y_shift, size)
                for single_img in x
            ]
        )
    elif len(x.shape) != 2:
        raise ValueError("Invalid array shape " + str(x.shape))

    backdoor = Image.open(backdoor_path)
    backdoored_input = Image.fromarray(np.copy(x), mode=mode)

    if size:
        backdoor = backdoor.resize(size).convert(mode=mode)

    backdoor_width, backdoor_height = backdoor.size
    orig_width, orig_height = backdoored_input.size

    if backdoor_width > orig_width or backdoor_height > orig_height:
        raise ValueError("Backdoor does not fit inside original image")

    if random:
        x_shift = np.random.randint(orig_width - backdoor_width)
        y_shift = np.random.randint(orig_height - backdoor_height)
    backdoored_input.paste(backdoor, box=(x_shift, y_shift))
    return np.array(backdoored_input)
