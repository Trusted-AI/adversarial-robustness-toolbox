# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This is a PyTorch implementation of the TRADES protocol.

| Paper link: https://proceedings.mlr.press/v97/zhang19p.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.defences.trainer.adversarial_trainer_trades import AdversarialTrainerTRADES
from art.estimators.classification.pytorch import PyTorchClassifier
from art.data_generators import DataGenerator
from art.attacks.attack import EvasionAttack
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
EPS = 1e-8


class AdversarialTrainerTRADESPyTorch(AdversarialTrainerTRADES):
    """
    Class performing adversarial training following TRADES protocol.

    | Paper link: https://proceedings.mlr.press/v97/zhang19p.html
    """

    def __init__(self, classifier: PyTorchClassifier, attack: EvasionAttack, beta: float):
        """
        Create an :class:`.AdversarialTrainerTRADESPyTorch` instance.

        :param classifier: Model to train adversarially.
        :param attack: attack to use for data augmentation in adversarial training
        :param beta: The scaling factor controlling tradeoff between clean loss and adversarial loss
        """
        super().__init__(classifier, attack, beta)
        self._classifier: PyTorchClassifier
        self._attack: EvasionAttack
        self._beta: float

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 128,
        nb_epochs: int = 20,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        **kwargs
    ):  # pylint: disable=W0221
        """
        Train a model adversarially with TRADES protocol.
        See class documentation for more information on the exact procedure.

        :param x: Training set.
        :param y: Labels for the training set.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for trainings.
        :param scheduler: Learning rate scheduler to run at the end of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """
        import torch

        logger.info("Performing adversarial training with TRADES protocol")
        # pylint: disable=W0212
        if (scheduler is not None) and (
            not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
        ):  # pylint: enable=W0212
            raise ValueError("Invalid Pytorch scheduler is provided for adversarial training.")

        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        logger.info("Adversarial Training TRADES")
        y = check_and_transform_label_format(y, nb_classes=self.classifier.nb_classes)

        if validation_data is not None:
            (x_test, y_test) = validation_data
            y_test = check_and_transform_label_format(y_test, nb_classes=self.classifier.nb_classes)

            x_preprocessed_test, y_preprocessed_test = self._classifier._apply_preprocessing(  # pylint: disable=W0212
                x_test, y_test, fit=True
            )

        for i_epoch in trange(nb_epochs, desc="Adversarial Training TRADES - Epochs"):
            # Shuffle the examples
            np.random.shuffle(ind)
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0

            for batch_id in range(nb_batches):
                # Create batch data
                x_batch = x[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                y_batch = y[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]]

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

            if scheduler:
                scheduler.step()

            train_time = time.time()

            # compute accuracy
            if validation_data is not None:
                output = np.argmax(self.predict(x_preprocessed_test), axis=1)
                nb_correct_pred = np.sum(output == np.argmax(y_preprocessed_test, axis=1))

                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc(tr): %.4f acc(val): %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                    nb_correct_pred / x_test.shape[0],
                )
            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )

    def fit_generator(
        self,
        generator: DataGenerator,
        nb_epochs: int = 20,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        **kwargs
    ):  # pylint: disable=W0221
        """
        Train a model adversarially with TRADES protocol using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :param nb_epochs: Number of epochs to use for trainings.
        :param scheduler: Learning rate scheduler to run at the end of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """
        import torch

        logger.info("Performing adversarial training with TRADES protocol")

        # pylint: disable=W0212
        if (scheduler is not None) and (
            not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
        ):  # pylint: enable=W0212
            raise ValueError("Invalid Pytorch scheduler is provided for adversarial training.")

        size = generator.size
        batch_size = generator.batch_size
        if size is not None:
            nb_batches = int(np.ceil(size / batch_size))
        else:
            raise ValueError("Size is None.")

        logger.info("Adversarial Training TRADES")

        for i_epoch in trange(nb_epochs, desc="Adversarial Training TRADES - Epochs"):
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0

            for batch_id in range(nb_batches):  # pylint: disable=W0612
                # Create batch data
                x_batch, y_batch = generator.get_batch()
                x_batch = x_batch.copy()

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

            if scheduler:
                scheduler.step()

            train_time = time.time()
            logger.info(
                "epoch: %s time(s): %.1f loss: %.4f acc: %.4f",
                i_epoch,
                train_time - start_time,
                train_loss / train_n,
                train_acc / train_n,
            )

    def _batch_process(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[float, float, float]:
        """
        Perform the operations of TRADES for a batch of data.
        See class documentation for more information on the exact procedure.

        :param x_batch: batch of x.
        :param y_batch: batch of y.
        :return: tuple containing batch data loss, batch data accuracy and number of samples in the batch
        """
        import torch
        from torch import nn
        import torch.nn.functional as F

        if self._classifier._optimizer is None:  # pylint: disable=W0212
            raise ValueError("Optimizer of classifier is currently None, but is required for adversarial training.")

        n = x_batch.shape[0]
        self._classifier._model.train(mode=False)  # pylint: disable=W0212
        x_batch_pert = self._attack.generate(x_batch, y=y_batch)

        # Apply preprocessing
        y_batch = check_and_transform_label_format(y_batch, nb_classes=self.classifier.nb_classes)

        x_preprocessed, y_preprocessed = self._classifier._apply_preprocessing(  # pylint: disable=W0212
            x_batch, y_batch, fit=True
        )
        x_preprocessed_pert, _ = self._classifier._apply_preprocessing(  # pylint: disable=W0212
            x_batch_pert, y_batch, fit=True
        )

        # Check label shape
        if self._classifier._reduce_labels:  # pylint: disable=W0212
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        i_batch = torch.from_numpy(x_preprocessed).to(self._classifier._device)  # pylint: disable=W0212
        i_batch_pert = torch.from_numpy(x_preprocessed_pert).to(self._classifier._device)  # pylint: disable=W0212
        o_batch = torch.from_numpy(y_preprocessed).to(self._classifier._device)  # pylint: disable=W0212

        self._classifier._model.train(mode=True)  # pylint: disable=W0212

        # Zero the parameter gradients
        self._classifier._optimizer.zero_grad()  # pylint: disable=W0212

        # Perform prediction
        model_outputs = self._classifier._model(i_batch)  # pylint: disable=W0212
        model_outputs_pert = self._classifier._model(i_batch_pert)  # pylint: disable=W0212

        # Form the loss function
        loss_clean = self._classifier._loss(model_outputs[-1], o_batch)  # pylint: disable=W0212
        loss_kl = (1.0 / n) * nn.KLDivLoss(reduction="sum")(
            F.log_softmax(model_outputs_pert[-1], dim=1), torch.clamp(F.softmax(model_outputs[-1], dim=1), min=EPS)
        )
        loss = loss_clean + self._beta * loss_kl
        loss.backward()

        self._classifier._optimizer.step()  # pylint: disable=W0212

        train_loss = loss.item() * o_batch.size(0)
        train_acc = (model_outputs_pert[0].max(1)[1] == o_batch).sum().item()
        train_n = o_batch.size(0)

        self._classifier._model.train(mode=False)  # pylint: disable=W0212

        return train_loss, train_acc, train_n
