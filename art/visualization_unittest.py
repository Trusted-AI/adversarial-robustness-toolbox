# MIT License
#
# Copyright (C) IBM Corporation 2018
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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os.path
import unittest

import numpy as np

from art import DATA_PATH
from art.utils import load_mnist, load_cifar10
from art.visualization import create_sprite, convert_to_rgb, save_image

logger = logging.getLogger('testLogger')


# python -m unittest discover art/ -p 'visualization_unittest.py'

class TestVisualization(unittest.TestCase):
    def test_save_image(self):
        (x, _), (_, _), _, _ = load_mnist(raw=True)

        f_name = 'image1.png'
        save_image(x[0], f_name)
        path = os.path.join(DATA_PATH, f_name)
        self.assertTrue(os.path.isfile(path))
        os.remove(path)

        f_name = 'image2.jpg'
        save_image(x[1], f_name)
        path = os.path.join(DATA_PATH, f_name)
        self.assertTrue(os.path.isfile(path))
        os.remove(path)

        folder = 'images123456'
        f_name_with_dir = os.path.join(folder, 'image3.png')
        save_image(x[3], f_name_with_dir)
        path = os.path.join(DATA_PATH, f_name_with_dir)
        self.assertTrue(os.path.isfile(path))
        os.remove(path)
        os.rmdir(os.path.split(path)[0])  # Remove also test folder

        folder = os.path.join('images123456', 'inner')
        f_name_with_dir = os.path.join(folder, 'image4.png')
        save_image(x[3], f_name_with_dir)
        path_nested = os.path.join(DATA_PATH, f_name_with_dir)
        self.assertTrue(os.path.isfile(path_nested))
        os.remove(path_nested)
        os.rmdir(os.path.split(path_nested)[0])  # Remove inner test folder
        os.rmdir(os.path.split(path)[0])  # Remove also test folder

    def test_convert_gray_to_rgb(self):
        # Get MNIST
        (x, _), (_, _), _, _ = load_mnist(raw=True)
        n = 100
        x = x[:n]

        # Test RGB
        x_rgb = convert_to_rgb(x)
        s_original = np.shape(x)
        s_new = np.shape(x_rgb)

        self.assertEqual(s_new[0], s_original[0])
        self.assertEqual(s_new[1], s_original[1])
        self.assertEqual(s_new[2], s_original[2])
        self.assertEqual(s_new[3], 3)  # Should have added 3 channels

    def test_sprites_gray(self):
        # Get MNIST
        (x, _), (_, _), _, _ = load_mnist(raw=True)
        n = 100
        x = x[:n]

        sprite = create_sprite(x)
        f_name = 'test_sprite_mnist.png'
        path = os.path.join(DATA_PATH, f_name)
        save_image(sprite, path)
        self.assertTrue(os.path.isfile(path))

        os.remove(path)  # Remove data added

    def test_sprites_color(self):
        (x, _), (_, _), _, _ = load_cifar10(raw=True)
        n = 500
        x = x[:n]

        sprite = create_sprite(x)
        f_name = 'test_cifar.jpg'
        path = os.path.join(DATA_PATH, f_name)
        save_image(sprite, path)
        self.assertTrue(os.path.isfile(path))

        os.remove(path)  # Remove data added


if __name__ == '__main__':
    unittest.main()
