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
This is a PyTorch implementation of the AWP protocol.

| Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time
from typing import Optional, Tuple, TYPE_CHECKING, List, Dict

from collections import OrderedDict
import numpy as np
from tqdm.auto import trange

from art.defences.trainer.adversarial_trainer_awp import AdversarialTrainerAWP
from art.estimators.classification.pytorch import PyTorchClassifier
from art.data_generators import DataGenerator
from art.attacks.attack import EvasionAttack

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
EPS = 1e-8


class AdversarialTrainerAWPPyTorch(AdversarialTrainerAWP):
    """
    Class performing adversarial training following AWP protocol.

    | Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    """

    def __init__(
        self,
        classifier: PyTorchClassifier,
        proxy_classifier: PyTorchClassifier,
        attack: EvasionAttack,
        mode: str,
        gamma: float,
        beta: float,
        warmup: int,
    ):
        """
        Create an :class:`.AdversarialTrainerAWPPyTorch` instance.

        :param classifier: Model to train adversarially.
        :param proxy_classifier: Model for adversarial weight perturbation.
        :param attack: attack to use for data augmentation in adversarial training.
        :param mode: mode determining the optimization objective of base adversarial training and weight perturbation
               step
        :param gamma: The scaling factor controlling norm of weight perturbation relative to model parameters' norm.
        :param beta: The scaling factor controlling tradeoff between clean loss and adversarial loss for TRADES protocol
        :param warmup: The number of epochs after which weight perturbation is applied
        """
        super().__init__(classifier, proxy_classifier, attack, mode, gamma, beta, warmup)
        self._classifier: PyTorchClassifier
        self._proxy_classifier: PyTorchClassifier
        self._attack: EvasionAttack
        self._mode: str
        self.gamma: float
        self._beta: float
        self._warmup: int
        self._apply_wp: bool

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 128,
        nb_epochs: int = 20,
        scheduler: "torch.optim.lr_scheduler._LRScheduler" = None,
        **kwargs,
    ):  # pylint: disable=W0221
        """
        Train a model adversarially with AWP protocol.
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

        logger.info("Performing adversarial training with AWP with %s protocol", self._mode)
        # pylint: disable=W0212
        if (scheduler is not None) and (
            not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
        ):  # pylint: enable=W0212
            raise ValueError("Invalid Pytorch scheduler is provided for adversarial training.")

        best_acc_adv_test = 0
        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        logger.info("Adversarial Training AWP with %s", self._mode)

        for i_epoch in trange(nb_epochs, desc=f"Adversarial Training AWP with {self._mode} - Epochs"):

            if i_epoch >= self._warmup:
                self._apply_wp = True
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
                (x_test, y_test) = validation_data
                output_clean = np.argmax(self.predict(x_test), axis=1)
                nb_correct_clean = np.sum(output_clean == np.argmax(y_test, axis=1))
                x_test_adv = self._attack.generate(x_test, y=y_test)
                output_adv = np.argmax(self.predict(x_test_adv), axis=1)
                nb_correct_adv = np.sum(output_adv == np.argmax(y_test, axis=1))

                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv (tr): %.4f acc-clean (val): %.4f acc-adv (val): %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                    nb_correct_clean / x_test.shape[0],
                    nb_correct_adv / x_test.shape[0],
                )

                # save last checkpoint
                if i_epoch + 1 == nb_epochs:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_{i_epoch}")

                # save best checkpoint
                if nb_correct_adv / x_test.shape[0] > best_acc_adv_test:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_best")
                    best_acc_adv_test = nb_correct_adv / x_test.shape[0]

            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )

    def fit_generator(
        self,
        generator: DataGenerator,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        nb_epochs: int = 20,
        scheduler: "torch.optim.lr_scheduler._LRScheduler" = None,
        **kwargs,
    ):  # pylint: disable=W0221
        """
        Train a model adversarially with AWP protocol using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param nb_epochs: Number of epochs to use for trainings.
        :param scheduler: Learning rate scheduler to run at the end of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """
        import torch

        logger.info("Performing adversarial training with AWP with %s protocol", self._mode)

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

        logger.info("Adversarial Training AWP with %s", self._mode)

        best_acc_adv_test = 0
        for i_epoch in trange(nb_epochs, desc=f"Adversarial Training AWP with {self._mode} - Epochs"):

            if i_epoch >= self._warmup:
                self._apply_wp = True

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

            # compute accuracy
            if validation_data is not None:
                (x_test, y_test) = validation_data
                output_clean = np.argmax(self.predict(x_test), axis=1)
                nb_correct_clean = np.sum(output_clean == np.argmax(y_test, axis=1))
                x_test_adv = self._attack.generate(x_test, y=y_test)
                output_adv = np.argmax(self.predict(x_test_adv), axis=1)
                nb_correct_adv = np.sum(output_adv == np.argmax(y_test, axis=1))

                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv (tr): %.4f acc-clean (val): %.4f acc-adv (val): %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                    nb_correct_clean / x_test.shape[0],
                    nb_correct_adv / x_test.shape[0],
                )
                # save last checkpoint
                if i_epoch + 1 == nb_epochs:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_{i_epoch}")

                # save best checkpoint
                if nb_correct_adv / x_test.shape[0] > best_acc_adv_test:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_best")
                    best_acc_adv_test = nb_correct_adv / x_test.shape[0]

            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )

    def _batch_process(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[float, float, float]:
        """
        Perform the operations of AWP for a batch of data.
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

        if self._proxy_classifier._optimizer is None:  # pylint: disable=W0212
            raise ValueError(
                "Optimizer of proxy classifier is currently None, but is required for adversarial training."
            )

        self._classifier._model.train(mode=False)  # pylint: disable=W0212
        x_batch_pert = self._attack.generate(x_batch, y=y_batch)

        # Apply preprocessing
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

        if self._apply_wp:
            w_perturb = self._weight_perturbation(x_batch=i_batch, x_batch_pert=i_batch_pert, y_batch=o_batch)
            list_keys = list(w_perturb.keys())
            self._modify_classifier(self._classifier, list_keys, w_perturb, op="add")

        # Zero the parameter gradients
        self._classifier._optimizer.zero_grad()  # pylint: disable=W0212

        if self._mode.lower() == "pgd":
            # Perform prediction
            model_outputs_pert = self._classifier._model(i_batch_pert)  # pylint: disable=W0212
            loss = self._classifier._loss(model_outputs_pert[-1], o_batch)  # pylint: disable=W0212

        elif self._mode.lower() == "trades":
            n = x_batch.shape[0]
            # Perform prediction
            model_outputs = self._classifier._model(i_batch)  # pylint: disable=W0212
            model_outputs_pert = self._classifier._model(i_batch_pert)  # pylint: disable=W0212

            # Form the loss function
            loss_clean = self._classifier._loss(model_outputs[-1], o_batch)  # pylint: disable=W0212
            loss_kl = (1.0 / n) * nn.KLDivLoss(reduction="sum")(
                F.log_softmax(model_outputs_pert[-1], dim=1), torch.clamp(F.softmax(model_outputs[-1], dim=1), min=EPS)
            )
            loss = loss_clean + self._beta * loss_kl

        else:
            raise ValueError(
                "Incorrect mode provided for base adversarial training. 'mode' must be among 'PGD' and 'TRADES'."
            )

        loss.backward()

        self._classifier._optimizer.step()  # pylint: disable=W0212

        if self._apply_wp:
            self._modify_classifier(self._classifier, list_keys, w_perturb, op="subtract")

        train_loss = loss.item() * o_batch.size(0)
        train_acc = (model_outputs_pert[0].max(1)[1] == o_batch).sum().item()
        train_n = o_batch.size(0)

        self._classifier._model.train(mode=False)  # pylint: disable=W0212

        return train_loss, train_acc, train_n

    def _weight_perturbation(
        self, x_batch: np.ndarray, x_batch_pert: np.ndarray, y_batch: np.ndarray
    ) -> Dict[str, "torch.Tensor"]:
        """
        Calculate wight perturbation for a batch of data.
        See class documentation for more information on the exact procedure.

        :param x_batch: batch of x.
        :param x_batch_pert: batch of x with perturbations.
        :param y_batch: batch of y.
        :return: dict containing names of classifier model's layers as keys and parameters as values
        """
        import torch
        from torch import nn
        import torch.nn.functional as F

        w_perturb = OrderedDict()
        params_dict, _ = self._calculate_model_params(self._classifier)
        list_keys = list(params_dict.keys())
        self._proxy_classifier._model.load_state_dict(self._classifier._model.state_dict())  # pylint: disable=W0212
        self._proxy_classifier._model.train(mode=True)  # pylint: disable=W0212

        if self._mode.lower() == "pgd":
            # Perform prediction
            model_outputs_pert = self._proxy_classifier._model(x_batch_pert)  # pylint: disable=W0212
            loss = -self._proxy_classifier._loss(model_outputs_pert[-1], y_batch)  # pylint: disable=W0212
        elif self._mode.lower() == "trades":
            n = x_batch.shape[0]
            # Perform prediction
            model_outputs = self._proxy_classifier._model(x_batch)  # pylint: disable=W0212
            model_outputs_pert = self._proxy_classifier._model(x_batch_pert)  # pylint: disable=W0212
            loss_clean = self._proxy_classifier._loss(model_outputs[-1], y_batch)  # pylint: disable=W0212
            loss_kl = (1.0 / n) * nn.KLDivLoss(reduction="sum")(
                F.log_softmax(model_outputs_pert[-1], dim=1), torch.clamp(F.softmax(model_outputs[-1], dim=1), min=EPS)
            )
            loss = -1.0 * (loss_clean + self._beta * loss_kl)

        else:
            raise ValueError(
                "Incorrect mode provided for base adversarial training. 'mode' must be among 'PGD' and 'TRADES'."
            )

        self._proxy_classifier._optimizer.zero_grad()  # pylint: disable=W0212
        loss.backward()
        self._proxy_classifier._optimizer.step()  # pylint: disable=W0212

        params_dict_proxy, _ = self._calculate_model_params(self._proxy_classifier)

        for name in list_keys:
            perturbation = params_dict_proxy[name]["param"] - params_dict[name]["param"]
            perturbation = perturbation.reshape(params_dict[name]["size"])
            scale = params_dict[name]["norm"] / (perturbation.norm() + EPS)
            w_perturb[name] = scale * perturbation

        return w_perturb

    # pylint: disable=R0201
    def _calculate_model_params(
        self, p_classifier: PyTorchClassifier
    ) -> Tuple[Dict[str, Dict[str, "torch.Tensor"]], "torch.Tensor"]:  # pylint: enable=R0201
        """

        :param p_classifier: model for awp protocol.
        :return: tuple with first element a dictionary with model parameters' names as keys and a nested dictionary
        as value. The nested dictionary contains model parameters, model parameters' size, model parameters' norms.
        The second element of tuple denotes norm of all model parameters
        """

        import torch

        params_dict = OrderedDict()
        list_params = []
        for name, param in p_classifier._model.state_dict().items():  # pylint: disable=W0212
            if len(param.size()) <= 1:
                continue
            if "weight" in name:
                temp_param = param.reshape(-1)
                list_params.append(temp_param)
                params_dict[name] = OrderedDict()
                params_dict[name]["param"] = temp_param
                params_dict[name]["size"] = param.size()
                params_dict[name]["norm"] = temp_param.norm()

        model_all_params = torch.cat(list_params)
        model_all_params_norm = model_all_params.norm()
        return params_dict, model_all_params_norm

    def _modify_classifier(
        self, p_classifier: PyTorchClassifier, list_keys: List[str], w_perturb: Dict[str, "torch.Tensor"], op: str
    ) -> None:
        """

        :param p_classifier: model for awp protocol.
        :param list_keys: list of model parameters' names
        :param w_perturb: dictionary containing model parameters' names as keys and model parameters as values
        :param op: controls whether weight perturbation will be added or subtracted from model parameters
        """

        import torch

        if op.lower() == "add":
            c_mult = 1.0
        elif op.lower() == "subtract":
            c_mult = -1.0
        else:
            raise ValueError("Incorrect op provided for weight perturbation. 'op' must be among 'add' and 'subtract'.")
        with torch.no_grad():
            for name, param in p_classifier._model.named_parameters():  # pylint: disable=W0212
                if name in list_keys:
                    param.add_(c_mult * self._gamma * w_perturb[name])
