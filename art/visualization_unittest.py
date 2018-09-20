from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import os.path
import numpy as np

from art.utils import load_mnist_raw, load_cifar10_raw
from art.visualization import create_sprite, convert_to_rgb, save_image
from art import DATA_PATH


# python -m unittest discover art/ -p 'visualization_unittest.py'

class TestVisualization(unittest.TestCase):

    def test_save_image(self):
        (x, _), (_, _), _, _ = load_mnist_raw()

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
        (x, _), (_, _), _, _ = load_mnist_raw()
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
        (x, _), (_, _), _, _ = load_mnist_raw()
        n = 100
        x = x[:n]

        f_name = 'test_sprite_mnist.png'
        create_sprite(x, f_name)
        path = os.path.join(DATA_PATH, f_name)
        self.assertTrue(os.path.isfile(path))
        # import matplotlib.pyplot as plt
        # plt.show(path)

        os.remove(path)  # Remove data added

    def test_sprites_color(self):
        (x, _), (_, _), _, _ = load_cifar10_raw()
        n = 500
        x = x[:n]

        f_name = 'test_cifar.jpg'
        create_sprite(x, f_name)
        path = os.path.join(DATA_PATH, f_name)
        self.assertTrue(os.path.isfile(path))
        # import matplotlib.pyplot as plt
        # plt.show(path)

        os.remove(path)  # Remove data added


if __name__ == '__main__':
    unittest.main()
