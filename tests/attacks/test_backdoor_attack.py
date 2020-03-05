# MIT License
#
# Copyright (C) IBM Corporation 2020
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
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.classifiers import KerasClassifier
from art.utils import preprocess, load_mnist
from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)

PP_POISON = 0.33


class TestBackdoorAttack(TestBase):
    """
    A unittest class for testing Backdoor Poisoning attack.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        master_seed(seed=1234)

        (cls.x_train_mnist_raw, cls.y_train_mnist_raw), (cls.x_test_mnist_raw, cls.y_test_mnist_raw), \
            cls._min, cls._max = load_mnist(raw=True)

        cls.n_train_raw = 7500
        cls.n_test_raw = 7500
        cls.x_train_mnist_raw = cls.x_train_mnist_raw[0:cls.n_train_raw]
        cls.y_train_mnist_raw = cls.y_train_mnist_raw[0:cls.n_train_raw]
        cls.x_test_mnist_raw = cls.x_test_mnist_raw[0:cls.n_test_raw]
        cls.y_test_mnist_raw = cls.y_test_mnist_raw[0:cls.n_test_raw]

    def setUp(self):
        master_seed(seed=1234)
        super().setUp()

    def get_keras_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.x_train_mnist.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        classifier = KerasClassifier(model=model, clip_values=(self._min, self._max))
        return classifier

    @staticmethod
    def poison_dataset(x_clean, y_clean, percent_poison, poison_func):
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

    def poison_func_1(self, x):
        max_val = np.max(self.x_train_mnist_raw)
        return add_pattern_bd(x, pixel_value=max_val)

    def poison_func_2(self, x):
        max_val = np.max(self.x_train_mnist_raw)
        return add_single_bd(x, pixel_value=max_val)

    def poison_func_3(self, x):
        return insert_image(x, backdoor_path='../../data/backdoors/flower.png', size=(5, 5), random=False,
                            x_shift=3, y_shift=3)

    def poison_func_4(self, x):
        return insert_image(x, backdoor_path='../../data/backdoors/post_it.png', size=(5, 5), random=True)

    def poison_func_5(self, x):
        return insert_image(x, backdoor_path='../../data/backdoors/flower.png', random=True)

    def test_backdoor_pattern(self):
        """
        Test the backdoor attack with a pattern-based perturbation
        """

        krc = self.get_keras_model()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = self.poison_dataset(self.x_train_mnist_raw,
                                                                                self.y_train_mnist_raw,
                                                                                PP_POISON, self.poison_func_1)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        # Add channel axis:
        x_train = np.expand_dims(x_train, axis=3)

        # Poison test data
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = self.poison_dataset(self.x_test_mnist_raw,
                                                                                         self.y_test_mnist_raw,
                                                                                         PP_POISON, self.poison_func_1)
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

        # check for high accuracy on clean samples
        self.assertGreater(clean_acc, 0.9)

        poison_x_test = x_test[is_poison_test]
        poison_y_test = y_test[is_poison_test]

        poison_preds = np.argmax(krc.predict(poison_x_test), axis=1)
        poison_correct = np.sum(poison_preds == np.argmax(poison_y_test, axis=1))
        poison_total = poison_y_test.shape[0]

        poison_acc = poison_correct / poison_total

        # check for targeted misclassification for poisoned samples
        self.assertGreater(poison_acc, 0.8)

    def test_backdoor_pixel(self):
        """
        Test the backdoor attack with a pixel-based perturbation
        """

        tfc = self.get_keras_model()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = self.poison_dataset(self.x_train_mnist_raw,
                                                                                self.y_train_mnist_raw,
                                                                                PP_POISON, self.poison_func_2)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        # Add channel axis:
        x_train = np.expand_dims(x_train, axis=3)

        # Poison test data
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = self.poison_dataset(self.x_test_mnist_raw,
                                                                                         self.y_test_mnist_raw,
                                                                                         PP_POISON, self.poison_func_2)
        x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
        # Add channel axis:
        x_test = np.expand_dims(x_test, axis=3)

        # Shuffle training data
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        tfc.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        clean_x_test = x_test[is_poison_test == 0]
        clean_y_test = y_test[is_poison_test == 0]

        clean_preds = np.argmax(tfc.predict(clean_x_test), axis=1)
        clean_correct = np.sum(clean_preds == np.argmax(clean_y_test, axis=1))
        clean_total = clean_y_test.shape[0]

        clean_acc = clean_correct / clean_total

        # check for high accuracy on clean samples
        self.assertGreater(clean_acc, 0.9)

        poison_x_test = x_test[is_poison_test]
        poison_y_test = y_test[is_poison_test]

        poison_preds = np.argmax(tfc.predict(poison_x_test), axis=1)
        poison_correct = np.sum(poison_preds == np.argmax(poison_y_test, axis=1))
        poison_total = poison_y_test.shape[0]

        poison_acc = poison_correct / poison_total

        # check for targeted misclassification for poisoned samples
        self.assertGreater(poison_acc, 0.8)

    def test_backdoor_image(self):
        """
        Test the backdoor attack with a image-based perturbation
        """
        tfc = self.get_keras_model()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = self.poison_dataset(self.x_train_mnist_raw,
                                                                                self.y_train_mnist_raw,
                                                                                PP_POISON, self.poison_func_3)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        # Add channel axis:
        x_train = np.expand_dims(x_train, axis=3)

        # Poison test data
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = self.poison_dataset(self.x_test_mnist_raw,
                                                                                         self.y_test_mnist_raw,
                                                                                         PP_POISON, self.poison_func_3)
        x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
        # Add channel axis:
        x_test = np.expand_dims(x_test, axis=3)

        # Shuffle training data
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        tfc.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        clean_x_test = x_test[is_poison_test == 0]
        clean_y_test = y_test[is_poison_test == 0]

        clean_preds = np.argmax(tfc.predict(clean_x_test), axis=1)
        clean_correct = np.sum(clean_preds == np.argmax(clean_y_test, axis=1))
        clean_total = clean_y_test.shape[0]

        clean_acc = clean_correct / clean_total

        # check for high accuracy on clean samples
        self.assertGreater(clean_acc, 0.9)

        poison_x_test = x_test[is_poison_test]
        poison_y_test = y_test[is_poison_test]

        poison_preds = np.argmax(tfc.predict(poison_x_test), axis=1)
        poison_correct = np.sum(poison_preds == np.argmax(poison_y_test, axis=1))
        poison_total = poison_y_test.shape[0]

        poison_acc = poison_correct / poison_total

        # check for targeted misclassification for poisoned samples
        self.assertGreater(poison_acc, 0.8)

    def test_multiple_perturbations(self):
        """
        Test using multiple perturbation functions in the same attack
        """

        tfc = self.get_keras_model()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = self.poison_dataset(self.x_train_mnist_raw,
                                                                                self.y_train_mnist_raw,
                                                                                PP_POISON, [self.poison_func_4,
                                                                                            self.poison_func_1])
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)

        # Add channel axis:
        x_train = np.expand_dims(x_train, axis=3)

        # Poison test data
        (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = self.poison_dataset(self.x_test_mnist_raw,
                                                                                         self.y_test_mnist_raw,
                                                                                         PP_POISON,
                                                                                         [self.poison_func_4,
                                                                                          self.poison_func_1])
        x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)

        # Add channel axis:
        x_test = np.expand_dims(x_test, axis=3)

        # Shuffle training data
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        tfc.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        clean_x_test = x_test[is_poison_test == 0]
        clean_y_test = y_test[is_poison_test == 0]

        clean_preds = np.argmax(tfc.predict(clean_x_test), axis=1)
        clean_correct = np.sum(clean_preds == np.argmax(clean_y_test, axis=1))
        clean_total = clean_y_test.shape[0]

        clean_acc = clean_correct / clean_total

        # check for high accuracy on clean samples
        self.assertGreater(clean_acc, 0.9)

        poison_x_test = x_test[is_poison_test]
        poison_y_test = y_test[is_poison_test]

        poison_preds = np.argmax(tfc.predict(poison_x_test), axis=1)
        poison_correct = np.sum(poison_preds == np.argmax(poison_y_test, axis=1))
        poison_total = poison_y_test.shape[0]

        poison_acc = poison_correct / poison_total

        # check for targeted misclassification for poisoned samples
        self.assertGreater(poison_acc, 0.8)

    def test_image_failure_modes(self):
        """
        Tests failure modes for image perturbation functions
        """

        backdoor_attack = PoisoningAttackBackdoor(self.poison_func_5)
        with self.assertRaises(ValueError) as context:
            backdoor_attack.poison(self.x_train_mnist_raw[0], y=self.y_train_mnist_raw[0]+1)

        self.assertIn('Backdoor does not fit inside original image', str(context.exception))

        with self.assertRaises(ValueError) as context:
            backdoor_attack.poison(np.zeros(5), y=self.y_train_mnist_raw[0] + 1)

        self.assertIn('Invalid array shape', str(context.exception))

        backdoor_attack = PoisoningAttackBackdoor(self.poison_func_1)
        with self.assertRaises(ValueError) as context:
            backdoor_attack.poison(np.zeros(5), y=self.y_train_mnist_raw[0] + 1)

        self.assertIn('Invalid array shape', str(context.exception))

        backdoor_attack = PoisoningAttackBackdoor(self.poison_func_2)
        with self.assertRaises(ValueError) as context:
            backdoor_attack.poison(np.zeros(5), y=self.y_train_mnist_raw[0] + 1)

        self.assertIn('Invalid array shape', str(context.exception))


if __name__ == '__main__':
    unittest.main()
