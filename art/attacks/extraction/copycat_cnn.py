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
"""
This module implements the copycat cnn attack `CopycatCNN`.

| Paper link: https://arxiv.org/abs/1806.05476
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks.attack import Attack
from art.classifiers.classifier import Classifier
from art.utils import to_categorical


logger = logging.getLogger(__name__)


class CopycatCNN(Attack):
    """
    Implementation of the copycat cnn attack from Jacson et al. (2018).

    | Paper link: https://arxiv.org/abs/1806.05476
    """
    attack_params = Attack.attack_params + ['batch_size', 'nb_epochs', 'nb_stolen']

    def __init__(self, classifier, batch_size=1, nb_epochs=10, nb_stolen=1):
        """
        Create a copycat cnn attack instance.

        :param classifier: A victim classifier.
        :type classifier: :class:`.Classifier`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param nb_stolen: Number of examples to be stolen.
        :type nb_stolen: `int`
        """
        super(CopycatCNN, self).__init__(classifier=classifier)

        params = {'batch_size': batch_size,
                  'nb_epochs': nb_epochs,
                  'nb_stolen': nb_stolen}
        self.set_params(**params)

    def generate(self, x, y=None, **kwargs):
        """
        Generate a thieved classifier.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Not used in this attack.
        :type y: `np.ndarray` or `None`
        :param thieved_classifier: A thieved classifier to be stolen.
        :type thieved_classifier: :class:`.Classifier`
        :return: The stolen classifier.
        :rtype: :class:`.Classifier`
        """
        # Warning to users if y is not None
        if y is not None:
            logger.warning("This attack does not use the provided label y.")

        # Check the size of the source input vs nb_stolen
        if x.shape[0] < self.nb_stolen:
            logger.warning("The size of the source input is smaller than the number of expected stolen examples.")

        # Check if there is a thieved classifier provided for training
        thieved_classifier = kwargs.get('thieved_classifier')
        if thieved_classifier is None or not isinstance(thieved_classifier, Classifier):
            raise ValueError('A thieved classifier is needed.')

        # Select data to attack
        selected_x = self._select_data(x)

        # Query the victim classifier
        fake_labels = self._query_label(selected_x)

        # Train the thieved classifier
        thieved_classifier.fit(x=selected_x, y=fake_labels)

        return thieved_classifier

    def _select_data(self, x):
        """
        Select data to attack.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :return: An array with the selected input to the victim classifier.
        :rtype: `np.ndarray`
        """
        nb_stolen = np.minimum(self.nb_stolen, x.shape[0])
        rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)

        return x[rnd_index]

    def _query_label(self, x):
        """
        Query the victim classifier.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :return: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :rtype: `np.ndarray`
        """
        labels = self.classifier.predict(x=x)
        labels = np.argmax(labels, axis=1)
        labels = to_categorical(labels=labels, nb_classes=self.classifier.nb_classes())

        return labels

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param nb_stolen: Number of examples to be stolen.
        :type nb_stolen: `int`
        """
        # Save attack-specific parameters
        super(CopycatCNN, self).set_params(**kwargs)

        if not isinstance(self.nb_epochs, (int, np.int)) or self.nb_epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")

        if not isinstance(self.nb_stolen, (int, np.int)) or self.nb_stolen <= 0:
            raise ValueError("The number of examples to be stolen must be a positive integer.")

        return True
