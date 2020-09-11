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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.defences.preprocessor import ThermometerEncoding

from tests.utils import master_seed

logger = logging.getLogger(__name__)


class TestThermometerEncoding(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(seed=1234)

    def test_channel_last(self):
        # Test data
        x = np.array(
            [
                [
                    [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                    [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                ],
                [
                    [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                    [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                ],
            ]
        )

        # Create an instance of ThermometerEncoding
        th_encoder = ThermometerEncoding(clip_values=(0, 1), num_space=4)

        # Preprocess
        x_preproc, _ = th_encoder(x)

        # Test
        self.assertEqual(x_preproc.shape, (2, 2, 3, 12))

        true_value = np.array(
            [
                [
                    [
                        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    ],
                    [
                        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    ],
                ],
                [
                    [
                        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    ],
                    [
                        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                    ],
                ],
            ]
        )
        self.assertTrue((x_preproc == true_value).all())

        # Create an instance of ThermometerEncoding
        th_encoder_scaled = ThermometerEncoding(clip_values=(-10, 10), num_space=4)
        x_preproc_scaled, _ = th_encoder_scaled(20 * x - 10)
        self.assertTrue((x_preproc_scaled == true_value).all())

    def test_channel_first(self):
        x = np.random.rand(5, 2, 28, 28)
        x_copy = x.copy()
        num_space = 5
        encoder = ThermometerEncoding(clip_values=(0, 1), num_space=num_space, channels_first=True)
        x_encoded, _ = encoder(x)
        self.assertTrue((x == x_copy).all())
        self.assertEqual(x_encoded.shape, (5, 10, 28, 28))

    def test_estimate_gradient(self):
        num_space = 5
        encoder = ThermometerEncoding(clip_values=(0, 1), num_space=num_space)
        encoder_cf = ThermometerEncoding(clip_values=(0, 1), num_space=num_space, channels_first=True)
        x = np.random.uniform(size=(5, 28, 28, 1))
        x_cf = np.transpose(x, (0, 3, 1, 2))
        grad = np.ones((5, 28, 28, num_space))
        grad_cf = np.transpose(grad, (0, 3, 1, 2))
        estimated_grads = encoder.estimate_gradient(grad=grad, x=x)
        estimated_grads_cf = encoder_cf.estimate_gradient(grad=grad_cf, x=x_cf)
        self.assertEqual(estimated_grads.shape, x.shape)
        self.assertEqual(estimated_grads_cf.shape, x_cf.shape)
        self.assertTrue((estimated_grads == np.transpose(estimated_grads_cf, (0, 2, 3, 1))).all())

    def test_feature_vectors(self):
        x = np.random.rand(10, 4)
        x_original = x.copy()
        num_space = 5
        encoder = ThermometerEncoding(clip_values=(0, 1), num_space=num_space, channels_first=True)
        x_encoded, _ = encoder(x)
        self.assertEqual(x_encoded.shape, (10, 20))
        # Check that x has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=0.00001)


if __name__ == "__main__":
    unittest.main()
