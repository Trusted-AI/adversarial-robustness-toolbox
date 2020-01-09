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

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import ExtractionAttack
from art.classifiers.classifier import Classifier
from art.utils import to_categorical


logger = logging.getLogger(__name__)


class CopycatCNN(ExtractionAttack):
    """
    Implementation of the Copycat CNN attack from Jacson et al. (2018).

    | Paper link: https://arxiv.org/abs/1806.05476
    """
    attack_params = ExtractionAttack.attack_params + ['batch_size_fit', 'batch_size_query', 'nb_epochs', 'nb_stolen']

    def __init__(self, classifier, batch_size_fit=1, batch_size_query=1, nb_epochs=10, nb_stolen=1):
        """
        Create a Copycat CNN attack instance.

        :param classifier: A victim classifier.
        :type classifier: :class:`.Classifier`
        :param batch_size_fit: Size of batches for fitting the thieved classifier.
        :type batch_size_fit: `int`
        :param batch_size_query: Size of batches for querying the victim classifier.
        :type batch_size_query: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param nb_stolen: Number of queries submitted to the victim classifier to steal it.
        :type nb_stolen: `int`
        """
        super(CopycatCNN, self).__init__(classifier=classifier)

        params = {'batch_size_fit': batch_size_fit,
                  'batch_size_query': batch_size_query,
                  'nb_epochs': nb_epochs,
                  'nb_stolen': nb_stolen}
        self.set_params(**params)

    def extract(self, x, y=None, **kwargs):
        """
        Extract a thieved classifier.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Not used in this attack.
        :type y: `np.ndarray` or `None`
        :param thieved_classifier: A classifier to be stolen, currently always trained on one-hot labels.
        :type thieved_classifier: :class:`.Classifier`
        :return: The stolen classifier.
        :rtype: :class:`.Classifier`
        """
        # Warning to users if y is not None
        if y is not None:
            logger.warning("This attack does not use the provided label y.")

        # Check the size of the source input vs nb_stolen
        if x.shape[0] < self.nb_stolen:
            logger.warning("The size of the source input is smaller than the expected number of queries submitted "
                           "to the victim classifier.")

        # Check if there is a thieved classifier provided for training
        thieved_classifier = kwargs.get('thieved_classifier')
        if thieved_classifier is None or not isinstance(thieved_classifier, Classifier):
            raise ValueError('A thieved classifier is needed.')

        # Select data to attack
        selected_x = self._select_data(x)

        # Query the victim classifier
        fake_labels = self._query_label(selected_x)

        # Train the thieved classifier
        thieved_classifier.fit(x=selected_x, y=fake_labels, batch_size=self.batch_size_fit, nb_epochs=self.nb_epochs)

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

        return x[rnd_index].astype(ART_NUMPY_DTYPE)

    def _query_label(self, x):
        """
        Query the victim classifier.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :return: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :rtype: `np.ndarray`
        """
        labels = self.classifier.predict(x=x, batch_size=self.batch_size_query)
        labels = np.argmax(labels, axis=1)
        labels = to_categorical(labels=labels, nb_classes=self.classifier.nb_classes())

        return labels

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param batch_size_fit: Size of batches for fitting the thieved classifier.
        :type batch_size_fit: `int`
        :param batch_size_query: Size of batches for querying the victim classifier.
        :type batch_size_query: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param nb_stolen: Number of queries submitted to the victim classifier to steal it.
        :type nb_stolen: `int`
        """
        # Save attack-specific parameters
        super(CopycatCNN, self).set_params(**kwargs)

        if not isinstance(self.batch_size_fit, (int, np.int)) or self.batch_size_fit <= 0:
            raise ValueError("The size of batches for fitting the thieved classifier must be a positive integer.")

        if not isinstance(self.batch_size_query, (int, np.int)) or self.batch_size_query <= 0:
            raise ValueError("The size of batches for querying the victim classifier must be a positive integer.")

        if not isinstance(self.nb_epochs, (int, np.int)) or self.nb_epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")

        if not isinstance(self.nb_stolen, (int, np.int)) or self.nb_stolen <= 0:
            raise ValueError("The number of queries submitted to the victim classifier must be a positive integer.")

        return True
