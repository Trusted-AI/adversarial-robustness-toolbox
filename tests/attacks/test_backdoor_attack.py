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
import unittest

import numpy as np

from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import preprocess
from tests.utils import TestBase, master_seed, get_classifier_kr
logger = logging.getLogger(__name__)


PP_POISON = 0.33


class TestBackdoorAttack(TestBase):
    """
    A unittest class for testing Backdoor Poisoning attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.n_train = 100
        cls.n_test = 100
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def setUp(self):
        super().setUpClass()
        master_seed(seed=1234)
        super().setUp()

    def poison_dataset(self, x_clean, y_clean, percent_poison, poison_func):
        max_val = np.max(x_clean)

        x_poison = np.copy(x_clean)
        y_poison = np.copy(y_clean)
        is_poison = np.zeros(np.shape(y_poison))

        sources = np.arange(10)  # 0, 1, 2, 3, ...
        targets = (np.arange(10) + 1) % 10  # 1, 2, 3, 4, ...
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            n_points_in_tgt = np.size(np.where(y_clean == tgt))
            num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
            src_imgs = x_clean[y_clean == src]

            n_points_in_src = np.shape(src_imgs)[0]
            indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

            imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
            backdoor_attack = PoisoningAttackBackdoor(poison_func)
            imgs_to_be_poisoned, poison_labels = backdoor_attack.poison(imgs_to_be_poisoned,
                                                                        y=np.ones(num_poison) * tgt)
            x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
            y_poison = np.append(y_poison, poison_labels, axis=0)
            is_poison = np.append(is_poison, np.ones(num_poison))

        is_poison = is_poison != 0

        return is_poison, x_poison, y_poison

    def test_backdoor_pattern(self):
        """
        Test the backdoor attack with a pattern-based perturbation
        :return:
        """
        max_val = np.max(self.x_train_mnist)

        def poison_func(x):
            return add_pattern_bd(x, pixel_value=max_val)
            # return add_single_bd(x, pixel_value=max_val)
            # return insert_image(x, backdoor_path='../data/backdoors/post_it.png', size=(5, 5))

        krc = get_classifier_kr()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = self.poison_dataset(self.x_train_mnist, self.y_train_mnist,
                                                                                PP_POISON, poison_func)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        # Add channel axis:
        x_train = np.expand_dims(x_train, axis=3)

        # Poison test data
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = self.poison_dataset(self.x_test_mnist,
                                                                                         self.y_test_mnist, PP_POISON,
                                                                                         poison_func)
        x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
        # Add channel axis:
        x_test = np.expand_dims(x_test, axis=3)

        # Shuffle training data
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        krc.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        clean_x_test = x_test[is_poison_test == 0]
        clean_y_test = y_test[is_poison_test == 0]

        clean_preds = np.argmax(krc.predict(clean_x_test), axis=1)
        clean_correct = np.sum(clean_preds == np.argmax(clean_y_test, axis=1))
        clean_total = clean_y_test.shape[0]

        clean_acc = clean_correct / clean_total

        self.assertGreater(clean_acc, 90)

        poison_x_test = x_test[is_poison_test]
        poison_y_test = y_test[is_poison_test]

        poison_preds = np.argmax(krc.predict(poison_x_test), axis=1)
        poison_correct = np.sum(poison_preds == np.argmax(poison_y_test, axis=1))
        poison_total = poison_y_test.shape[0]

        poison_acc = poison_correct / poison_total
        self.assertGreater(poison_acc, 0.8)

    # TODO: just copy and paste from here
    def test_backdoor_pixel(self):
        self.assertEqual(1, 1)

    def test_backdoor_image(self):
        self.assertEqual(1, 1)

    def test_multiple_perturbations(self):
        import os
        print(os.getcwd())
        self.assertEqual(1, 1)

    def test_failure_image(self):
        # TODO: test for backdoor images that are too big
        # TODO: test for invalid array shapes
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
