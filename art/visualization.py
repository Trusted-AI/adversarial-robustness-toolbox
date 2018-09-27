"""
Module providing visualization functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from art import DATA_PATH

import numpy as np
import os.path


def create_sprite(images):
    """
    Creates a sprite of provided images

    :param images: Images to construct the sprite.
    :type images: `np.array`
    :return:
    """

    shape = np.shape(images)

    if len(shape) < 3 or len(shape) > 4:
        raise ValueError(
            "Images provided for sprite have wrong dimensions " + str(len(shape)))

    if len(shape) == 3:
        # Check to see if it's mnist type of images and add axis to show image is gray-scale
        images = np.expand_dims(images, axis=3)
        shape = np.shape(images)

    # Change black and white images to RGB
    if shape[3] == 1:
        images = convert_to_rgb(images)

    n = int(np.ceil(np.sqrt(images.shape[0])))
    padding = ((0, n ** 2 - images.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode='constant',
                    constant_values=0)
    # Tile the individual thumbnails into an image.
    images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3)
                                                                 + tuple(range(4, images.ndim + 1)))
    images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])
    sprite = (images * 255).astype(np.uint8)

    return sprite


def convert_to_rgb(images):
    """
    Converts gray scale images to RGB. It changes NxHxWx1 to a NxHxWx3 tensor,
    where N is the number of figures, H is the high and W the width

    :param images: Gray scale images NxHxWx1
    :return: rgb_images
    """
    s = np.shape(images)
    if not ((len(s) == 4 and s[-1] == 1) or len(s) == 3):
        raise ValueError('Unexpected shape for grayscale images:' + str(s))

    if s[-1] == 1:
        # Squeeze channel axis if it exists
        rgb_images = np.squeeze(images, axis=-1)
    else:
        rgb_images = images
    rgb_images = np.stack((rgb_images,) * 3, axis=-1)

    return rgb_images


def save_image(image, f_name):
    """
    Saves image into a file inside DATA_PATH with the name f_name

    :param image: Image to be saved
    :type image: `np.ndarray`
    :param f_name: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png
    :type f_name: `str`
    :return: `None`
    """
    file_name = os.path.join(DATA_PATH, f_name)
    if not os.path.exists(os.path.split(file_name)[0]):
        os.makedirs(os.path.split(file_name)[0])

    import scipy.misc
    scipy.misc.toimage(image, cmin=0.0, cmax=int(np.max(image))).save(file_name)
    print("Image saved to ", file_name)
