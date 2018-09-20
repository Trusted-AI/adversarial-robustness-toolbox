"""
Module providing visualization functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from art import DATA_PATH

import numpy as np
import cv2
import os.path


def create_sprite(images, f_name):
    """
    Creates a sprite of provided images and saves it in DATA_PATH with file name f_name

    :param images: Images to construct the sprite.
    :type images: `np.array`
    :param f_name: File name of the sprite containing extension e.g., my_sprite.jpg, my_sprite.png, images/my_sprite.png
    :type f_name: `str`
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

    # Save image
    save_image(sprite, f_name)
    return


def convert_to_rgb(images):
    """
    Converts gray scale images to RGB. It changes NxHxWx1 to a NxHxWx3 tensor,
    where N is the number of figures, H is the high and W the width

    :param images: Gray scale images NxHxWx1
    :return: rgb_images
    """
    s = np.shape(images)

    rgb_images = np.zeros(shape=(s[0], s[1], s[2], 3))
    for i, img in enumerate(images):
        new_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rgb_images[i] = new_image

    return rgb_images


def save_image(image, f_name):
    """
    Saves image into a file inside DATA_PATH with the name f_name

    :param image: Image to be saved
    :param f_name: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png
    :type f_name: `str`
    :return:
    """
    file_name = os.path.join(DATA_PATH, f_name)
    if not os.path.exists(os.path.split(file_name)[0]):
        os.makedirs(os.path.split(file_name)[0])

    cv2.imwrite(file_name, image)
    print("Image saved to ", file_name)
