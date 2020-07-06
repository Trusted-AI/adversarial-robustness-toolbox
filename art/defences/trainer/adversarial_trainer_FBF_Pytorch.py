# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This is a PyTorch implementation of the Fast is better than free protocol.

| Paper link: https://openreview.net/forum?id=BJx040EFvH
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.defences.trainer.adversarial_trainer_FBF import AdversarialTrainerFBF
from art.utils import random_sphere

logger = logging.getLogger(__name__)


class AdversarialTrainerFBFPyTorch(AdversarialTrainerFBF):
    """
    Class performing adversarial training following Fast is Better Than Free protocol.

    | Paper link: https://openreview.net/forum?id=BJx040EFvH

    | The effectiveness of this protoocl is found to be sensitive to the use of techniques like
        data augmentation, gradient clipping and learning rate schedules. Optionally, the use of
        mixed precision arithmetic operation via apex library can significantly reduce the training
        time making this one of the fastest adversarial training protocol.
    """

    def __init__(self, classifier, eps=8, use_amp=False, **kwargs):
        """
        Create an :class:`.AdversarialTrainerFBFPyTorch` instance.

        :param classifier: Model to train adversarially.
        :type classifier: :class:`.Classifier`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param use_amp: Boolean that decides if apex should be used for mixed precision arithmantic during training
        :type use_amp: `bool`

        """
        super().__init__(classifier, eps, **kwargs)
        self._use_amp = use_amp

    def fit(self, x, y, validation_data=None, batch_size=128, nb_epochs=20, **kwargs):
        """
        Train a model adversarially with FBF protocol.
        See class documentation for more information on the exact procedure.

        :param x: Training set.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param validation_data: Tuple consisting of validation data
        :type validation_data: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        :type kwargs: `dict`
        :return: `None`
        """
        logger.info("Performing adversarial training with Fast is better than Free protocol")

        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        def lr_schedule(t):
            return np.interp([t], [0, nb_epochs * 2 // 5, nb_epochs], [0, 0.21, 0])[0]

        for i_epoch in range(nb_epochs):
            logger.info("Adversarial training FBF epoch %i/%i", i_epoch, nb_epochs)

            # Shuffle the examples
            np.random.shuffle(ind)
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0

            for batch_id in range(nb_batches):
                lr = lr_schedule(i_epoch + (batch_id + 1) / nb_batches)

                # Create batch data
                x_batch = x[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                y_batch = y[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]]

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch, lr)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

            train_time = time.time()

            # compute accuracy
            if validation_data is not None:
                (x_test, y_test) = validation_data
                output = np.argmax(self.predict(x_test), axis=1)
                nb_correct_pred = np.sum(output == np.argmax(y_test, axis=1))
                logger.info(
                    "{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                        i_epoch,
                        train_time - start_time,
                        lr,
                        train_loss / train_n,
                        train_acc / train_n,
                        nb_correct_pred / x_test.shape[0],
                    )
                )
            else:
                logger.info(
                    "{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                        i_epoch, train_time - start_time, lr, train_loss / train_n, train_acc / train_n
                    )
                )

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Train a model adversarially with FBF protocol using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        :type kwargs: `dict`
        :return: `None`
        """
        logger.info("Performing adversarial training with Fast is better than Free protocol")
        size = generator.size
        batch_size = generator.batch_size
        nb_batches = int(np.ceil(size / batch_size))

        def lr_schedule(t):
            return np.interp([t], [0, nb_epochs * 2 // 5, nb_epochs], [0, 0.21, 0])[0]

        for i_epoch in range(nb_epochs):
            logger.info("Adversarial training FBF epoch %i/%i", i_epoch, nb_epochs)
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0

            for batch_id in range(nb_batches):
                lr = lr_schedule(i_epoch + (batch_id + 1) / nb_batches)

                # Create batch data
                x_batch, y_batch = generator.get_batch()
                x_batch = x_batch.copy()

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch, lr)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

            train_time = time.time()
            logger.info(
                "{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                    i_epoch, train_time - start_time, lr, train_loss / train_n, train_acc / train_n
                )
            )

    def _batch_process(self, x_batch, y_batch, lr):
        """
        Perform the operations of FBF for a batch of data.
        See class documentation for more information on the exact procedure.

        :param x_batch: batch of x.
        :type x_batch: `np.ndarray`
        :param y_batch: batch of y.
        :type y_batch: `np.ndarray`
        :param lr: learning rate for the optimisation step.
        :type lr: `float`
        :return: `(float, float, float)`
        """
        import torch

        n = x_batch.shape[0]
        m = np.prod(x_batch.shape[1:])
        delta = random_sphere(n, m, self._eps, np.inf).reshape(x_batch.shape).astype(ART_NUMPY_DTYPE)
        delta_grad = self._classifier.loss_gradient(x_batch + delta, y_batch)
        delta = np.clip(delta + 1.25 * self._eps * np.sign(delta_grad), -self._eps, +self._eps)
        x_batch_pert = np.clip(x_batch + delta, self._classifier.clip_values[0], self._classifier.clip_values[1])

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._classifier._apply_preprocessing(x_batch_pert, y_batch, fit=True)

        # Check label shape
        if self._classifier._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        i_batch = torch.from_numpy(x_preprocessed).to(self._classifier._device)
        o_batch = torch.from_numpy(y_preprocessed).to(self._classifier._device)

        # Zero the parameter gradients
        self._classifier._optimizer.zero_grad()

        # Perform prediction
        model_outputs = self._classifier._model(i_batch)

        # Form the loss function
        loss = self._classifier._loss(model_outputs[-1], o_batch)

        self._classifier._optimizer.param_groups[0].update(lr=lr)

        # Actual training
        if self._use_amp:
            import apex.amp as amp

            with amp.scale_loss(loss, self._classifier._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(self._classifier._model.parameters(), 0.5)
        self._classifier._optimizer.step()

        train_loss = loss.item() * o_batch.size(0)
        train_acc = (model_outputs[0].max(1)[1] == o_batch).sum().item()
        train_n = o_batch.size(0)

        return train_loss, train_acc, train_n
