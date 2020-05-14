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
| Paper link: https://openreview.net/forum?id=BJx040EFvH
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.defences.trainer.trainer import Trainer
import random
from art.utils import random_sphere

logger = logging.getLogger(__name__)


class AdversarialTrainerFBF(Trainer):
    """
    | Paper link: https://openreview.net/forum?id=BJx040EFvH
    """

    def __init__(self, classifier, eps=8):
        """
        Create an :class:`.AdversarialTrainer` instance.

        :param classifier: Model to train adversarially.
        :type classifier: :class:`.Classifier`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`

        """
        self.eps = eps
        self.classifier = classifier
        # Setting up adversary and perform adversarial training:


    def fit(self, x, y, validation_data=None, batch_size=128, nb_epochs=20, **kwargs):
        """
        Train a model adversarially. See class documentation for more information on the exact procedure.

        :param x: Training set.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :type kwargs: `dict`
        :return: `None`
        """
        import torch
        import torch.nn as nn

        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        lr_schedule = lambda t: np.interp([t], [0, nb_epochs * 2 // 5, nb_epochs], [0, 0.21, 0])[0]

        for i_epoch in range(nb_epochs):
            logger.info("Adversarial training FBF epoch %i/%i", i_epoch, nb_epochs)

            # Shuffle the examples
            np.random.shuffle(ind)

            for batch_id in range(nb_batches):
                lr = lr_schedule(i_epoch + (batch_id + 1) / nb_batches)
                self.classifier._optimizer.param_groups[0].update(lr=lr)
                # Create batch data
                x_batch = x[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                y_batch = y[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]]

                # adv_ids = list(range(x_batch.shape[0]))
                # np.random.shuffle(adv_ids)
                # generate delta as per

                n = x_batch.shape[0]
                m = np.prod(x_batch.shape[1:])
                delta = random_sphere(n, m, self.eps, np.inf).reshape(x_batch.shape).astype(ART_NUMPY_DTYPE)
                delta_grad = self.classifier.loss_gradient(x_batch + delta,y_batch)
                delta = np.clip(delta + 1.25*self.eps*np.sign(delta_grad), -self.eps, +self.eps)
                x_batch_pert = np.clip(x_batch+delta,self.classifier.clip_values[0], self.classifier.clip_values[1])

                # Fit batch
                # self.classifier.fit(x_batch_pert, y_batch, nb_epochs=1, batch_size=x_batch.shape[0], **kwargs)

                # Apply preprocessing
                x_preprocessed, y_preprocessed = self.classifier._apply_preprocessing(x_batch_pert, y_batch, fit=True)

                # Check label shape
                if self.classifier._reduce_labels:
                    y_preprocessed = np.argmax(y_preprocessed, axis=1)

                i_batch = torch.from_numpy(x_preprocessed).to(
                    self.classifier._device)
                o_batch = torch.from_numpy(y_preprocessed).to(
                    self.classifier._device)

                # Zero the parameter gradients
                self.classifier._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self.classifier._model(i_batch)

                # Form the loss function
                loss = self.classifier._loss(model_outputs[-1], o_batch)

                # Actual training
                loss.backward()
                nn.utils.clip_grad_norm_(self.classifier._model.parameters(), 0.5)
                self.classifier._optimizer.step()

    def predict(self, x, **kwargs):
        """
        Perform prediction using the adversarially trained classifier.

        :param x: Test set.
        :type x: `np.ndarray`
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :type kwargs: `dict`
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)
