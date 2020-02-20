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
"""
This module implements the distillation defence `Distillation`.

| Paper link: https://arxiv.org/abs/1511.04508
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.defences.transform.transformer import Transformer

logger = logging.getLogger(__name__)


class Distillation(Transformer):
    """
    This module implements the distillation defence `Distillation`.
    Implement the distillation defence approach.

    | Paper link: https://arxiv.org/abs/1511.04508
    """

    params = ["batch_size", "nb_epochs"]

    def __init__(self, classifier, batch_size=128, nb_epochs=10):
        """
        Create an instance of the distillation defence.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        """
        super(Distillation, self).__init__(classifier=classifier)

        kwargs = {
            "batch_size": batch_size,
            "nb_epochs": nb_epochs,
        }
        self.set_params(**kwargs)




    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        """
        # Save defence-specific parameters
        super(Distillation, self).set_params(**kwargs)

        if not isinstance(self.batch_size, (int, np.int)) or self.batch_size <= 0:
            raise ValueError("The size of batches must be a positive integer.")

        if not isinstance(self.nb_epochs, (int, np.int)) or self.nb_epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")

        return True
