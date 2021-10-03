# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
Module providing visualization functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os.path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

from art import config

if TYPE_CHECKING:
    import matplotlib

logger = logging.getLogger(__name__)


def create_sprite(images: np.ndarray) -> np.ndarray:
    """
    Creates a sprite of provided images.

    :param images: Images to construct the sprite.
    :return: An image array containing the sprite.
    """
    shape = np.shape(images)

    if len(shape) < 3 or len(shape) > 4:
        raise ValueError("Images provided for sprite have wrong dimensions " + str(len(shape)))

    if len(shape) == 3:
        # Check to see if it's MNIST type of images and add axis to show image is gray-scale
        images = np.expand_dims(images, axis=3)
        shape = np.shape(images)

    # Change black and white images to RGB
    if shape[3] == 1:
        images = convert_to_rgb(images)

    n = int(np.ceil(np.sqrt(images.shape[0])))
    padding = ((0, n ** 2 - images.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode="constant", constant_values=0)

    # Tile the individual thumbnails into an image
    images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
    images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])
    sprite = (images * 255).astype(np.uint8)

    return sprite


def convert_to_rgb(images: np.ndarray) -> np.ndarray:
    """
    Converts grayscale images to RGB. It changes NxHxWx1 to a NxHxWx3 array, where N is the number of figures,
    H is the high and W the width.

    :param images: Grayscale images of shape (NxHxWx1).
    :return: Images in RGB format of shape (NxHxWx3).
    """
    dims = np.shape(images)
    if not ((len(dims) == 4 and dims[-1] == 1) or len(dims) == 3):
        raise ValueError("Unexpected shape for grayscale images:" + str(dims))

    if dims[-1] == 1:
        # Squeeze channel axis if it exists
        rgb_images = np.squeeze(images, axis=-1)
    else:
        rgb_images = images
    rgb_images = np.stack((rgb_images,) * 3, axis=-1)

    return rgb_images


def save_image(image_array: np.ndarray, f_name: str) -> None:
    """
    Saves image into a file inside `ART_DATA_PATH` with the name `f_name`.

    :param image_array: Image to be saved.
    :param f_name: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png.
    """
    file_name = os.path.join(config.ART_DATA_PATH, f_name)
    folder = os.path.split(file_name)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    image = Image.fromarray(image_array)
    image.save(file_name)
    logger.info("Image saved to %s.", file_name)


def plot_3d(
    points: np.ndarray,
    labels: List[int],
    colors: Optional[List[str]] = None,
    save: bool = True,
    f_name: str = "",
) -> "matplotlib.figure.Figure":  # pragma: no cover
    """
    Generates a 3-D plot in of the provided points where the labels define the color that will be used to color each
    data point. Concretely, the color of points[i] is defined by colors(labels[i]). Thus, there should be as many labels
     as colors.

    :param points: arrays with 3-D coordinates of the plots to be plotted.
    :param labels: array of integers that determines the color used in the plot for the data point.
        Need to start from 0 and be sequential from there on.
    :param colors: Optional argument to specify colors to be used in the plot. If provided, this array should contain
    as many colors as labels.
    :param save:  When set to True, saves image into a file inside `ART_DATA_PATH` with the name `f_name`.
    :param f_name: Name used to save the file when save is set to True.
    :return: A figure object.
    """
    # Disable warnings of unused import because all imports in this block are required
    # pylint: disable=W0611
    # import matplotlib  # lgtm [py/repeated-import]
    import matplotlib.pyplot as plt  # lgtm [py/repeated-import]

    # from mpl_toolkits import mplot3d

    if colors is None:  # pragma: no cover
        colors = []
        for i in range(len(np.unique(labels))):
            colors.append("C" + str(i))
    else:
        if len(colors) != len(np.unique(labels)):
            raise ValueError("The amount of provided colors should match the number of labels in the 3pd plot.")

    fig = plt.figure()
    axis = plt.axes(projection="3d")

    for i, coord in enumerate(points):
        try:
            color_point = labels[i]
            axis.scatter3D(coord[0], coord[1], coord[2], color=colors[color_point])
        except IndexError:
            raise ValueError(
                "Labels outside the range. Should start from zero and be sequential there after"
            ) from IndexError
    if save:  # pragma: no cover
        file_name = os.path.realpath(os.path.join(config.ART_DATA_PATH, f_name))
        folder = os.path.split(file_name)[0]

        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(file_name, bbox_inches="tight")
        logger.info("3d-plot saved to %s.", file_name)

    return fig
