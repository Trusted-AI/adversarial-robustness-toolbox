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
This is a PyTorch implementation of the Oracle Aligned Adversarial Training (OAAT) protocol
for adversarial training for defence against larger perturbations.

| Paper link: https://link.springer.com/chapter/10.1007/978-3-031-20065-6_18
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
import logging
import os
import time
from typing import Optional, Tuple, TYPE_CHECKING, List, Dict, Union

import six
import numpy as np
from tqdm.auto import trange

from art import config
from art.defences.trainer.adversarial_trainer_oaat import AdversarialTrainerOAAT
from art.estimators.classification.pytorch import PyTorchClassifier
from art.data_generators import DataGenerator
from art.attacks.attack import EvasionAttack
from art.utils import check_and_transform_label_format, random_sphere
from art.config import ART_NUMPY_DTYPE

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)
EPS = 1e-8  # small value required for avoiding division by zero and for KLDivLoss to make probability vector non-zero


class AdversarialTrainerOAATPyTorch(AdversarialTrainerOAAT):
    """
    Class performing adversarial training following Oracle Aligned Adversarial Training (OAAT) protocol.

    | Paper link: https://link.springer.com/chapter/10.1007/978-3-031-20065-6_18
    """

    def __init__(
        self,
        classifier: PyTorchClassifier,
        proxy_classifier: PyTorchClassifier,
        lpips_classifier: PyTorchClassifier,
        list_avg_models: List[PyTorchClassifier],
        attack: EvasionAttack,
        train_params: dict,
    ):
        """
        Create an :class:`.AdversarialTrainerOAATPyTorch` instance.

        :param classifier: Model to train adversarially.
        :param proxy_classifier: Model for adversarial weight perturbation.
        :param lpips_classifier: Weight averaging model for calculating activations.
        :param list_avg_models: list of models for weight averaging.
        :param attack: attack to use for data augmentation in adversarial training.
        :param train_params: training parameters' dictionary related to adversarial training
        """
        super().__init__(classifier, proxy_classifier, lpips_classifier, list_avg_models, attack, train_params)
        self._classifier: PyTorchClassifier
        self._proxy_classifier: PyTorchClassifier
        self._lpips_classifier: PyTorchClassifier
        self._list_avg_models: List[PyTorchClassifier]
        self._attack: EvasionAttack
        self._train_params: dict
        self._apply_wp: bool
        self._apply_lpips_pert: bool

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 128,
        nb_epochs: int = 20,
        **kwargs,
    ):  # pylint: disable=W0221
        """
        Train a model adversarially with OAAT protocol.
        See class documentation for more information on the exact procedure.

        :param x: Training set.
        :param y: Labels for the training set.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for trainings.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """
        import torch

        logger.info("Performing adversarial training with OAAT protocol")

        best_acc_adv_test = 0
        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        logger.info("Adversarial Training OAAT")

        if (len(self._train_params["list_swa_epoch"]) != len(self._train_params["list_swa_tau"])) or (
            len(self._train_params["list_swa_epoch"]) != len(self._list_avg_models)
        ):
            raise ValueError(
                "number of elements of list_swa_epoch must be same as of  " "list_swa_tau and list_avg_models."
            )

        beta_init = self._train_params["beta"]
        y = check_and_transform_label_format(y, nb_classes=self.classifier.nb_classes)

        for i_epoch in trange(nb_epochs, desc="Adversarial Training OAAT - Epochs"):

            if i_epoch >= self._train_params["awp_warmup"]:
                self._apply_wp = True

            if i_epoch == (int(3 * nb_epochs / 4)):
                self._classifier._optimizer = torch.optim.SGD(  # pylint: disable=W0212
                    self._classifier.model.parameters(),
                    lr=self._train_params["lr"],
                    momentum=self._train_params["momentum"],
                    weight_decay=self._train_params["weight_decay"],
                )

            if isinstance(self._classifier.optimizer, torch.optim.SGD):

                if self._train_params["oaat_warmup"] == 1:
                    if i_epoch < 10:
                        self.update_learning_rate(self._classifier.optimizer, i_epoch, nb_epochs, lr_schedule="linear")
                    else:
                        self.update_learning_rate(
                            self._classifier.optimizer,
                            i_epoch,
                            nb_epochs,
                            lr_schedule=self._train_params["lr_schedule"],
                        )
                else:

                    self.update_learning_rate(
                        self._classifier.optimizer, i_epoch, nb_epochs, lr_schedule=self._train_params["lr_schedule"]
                    )

            if i_epoch > (nb_epochs // 4):
                self._train_params["i_epsilon"] = i_epoch * self._train_params["epsilon"] / nb_epochs

            if self._train_params["i_epsilon"] > self._train_params["alternate_iter_eps"]:
                self._apply_lpips_pert = True

                load_swa_model_tau = self._train_params["load_swa_model_tau"]
                file_name = f"oaat_swa_tau_{load_swa_model_tau}.model"
                if self._train_params["models_path"] is None:
                    full_path_load = os.path.join(config.ART_DATA_PATH, file_name)
                else:
                    full_path_load = os.path.join(self._train_params["models_path"], file_name)
                if os.path.isfile(full_path_load):
                    self._lpips_classifier._model._model.load_state_dict(  # pylint: disable=W0212
                        torch.load(full_path_load)
                    )
                else:
                    raise ValueError("Invalid path/file for weight average model is provided for adversarial training.")

                self._lpips_classifier.model.train(mode=False)
                self._lpips_classifier.model.to(self._classifier.device)

                self._train_params["alpha"] = self._train_params["alpha"] - self._train_params["mixup_alpha"] / (
                    nb_epochs - int(3 * nb_epochs / 4) + 1
                )
                self._train_params["i_lpips_weight"] = (
                    self._train_params["lpips_weight"]
                    * (i_epoch - int(3 * nb_epochs / 4))
                    / (nb_epochs - int(3 * nb_epochs / 4))
                )
                self._train_params["beta"] = beta_init + (self._train_params["beta_final"] - beta_init) * (
                    i_epoch - int(3 * nb_epochs / 4)
                ) / (nb_epochs - int(3 * nb_epochs / 4))

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

                _train_loss, _train_acc, _train_n = self._batch_process(i_epoch, nb_epochs, batch_id, x_batch, y_batch)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

            train_time = time.time()

            # compute accuracy
            if validation_data is not None:
                (x_test, y_test) = validation_data
                y_test = check_and_transform_label_format(y_test, nb_classes=self.classifier.nb_classes)
                # pylint: disable=W0212
                x_preprocessed_test, y_preprocessed_test = self._classifier._apply_preprocessing(
                    x_test,
                    y_test,
                    fit=True,
                )
                # pylint: enable=W0212
                output_clean = np.argmax(self.predict(x_preprocessed_test), axis=1)
                nb_correct_clean = np.sum(output_clean == np.argmax(y_preprocessed_test, axis=1))
                self._attack.set_params(
                    eps=self._train_params["epsilon"],
                    eps_step=self._train_params["epsilon"] / 4.0,
                    max_iter=self._train_params["max_iter"],
                )
                x_test_adv = self._attack.generate(x_preprocessed_test, y=y_preprocessed_test)
                # pylint: disable=W0212
                x_preprocessed_test_adv, y_preprocessed_test = self._classifier._apply_preprocessing(
                    x_test_adv,
                    y_test,
                    fit=True,
                )
                # pylint: enable=W0212
                output_adv = np.argmax(self.predict(x_preprocessed_test_adv), axis=1)
                nb_correct_adv = np.sum(output_adv == np.argmax(y_preprocessed_test, axis=1))

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
                    self._classifier.save(filename=f"oaat_epoch_{i_epoch}", path=self._train_params["models_path"])

                # save best checkpoint
                if nb_correct_adv / x_test.shape[0] > best_acc_adv_test:
                    self._classifier.save(filename="oaat_epoch_best", path=self._train_params["models_path"])
                    best_acc_adv_test = nb_correct_adv / x_test.shape[0]

            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )

            if i_epoch > self._train_params["swa_save_epoch"] - 1:
                for epoch_swa, tau_val, p_classifier in zip(
                    self._train_params["list_swa_epoch"], self._train_params["list_swa_tau"], self._list_avg_models
                ):
                    if epoch_swa <= i_epoch:
                        file_name = f"oaat_swa_tau_{tau_val}.model"
                        if self._train_params["models_path"] is None:
                            full_path = os.path.join(config.ART_DATA_PATH, file_name)
                        else:
                            full_path = os.path.join(self._train_params["models_path"], file_name)

                        folder = os.path.split(full_path)[0]
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        # pylint: disable=W0212
                        # disable pylint because access to _modules required
                        torch.save(p_classifier._model._model.state_dict(), full_path)

    def fit_generator(
        self,
        generator: DataGenerator,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        nb_epochs: int = 20,
        **kwargs,
    ):  # pylint: disable=W0221
        """
        Train a model adversarially with OAAT protocol using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param nb_epochs: Number of epochs to use for trainings.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """
        import torch

        logger.info("Performing adversarial training with OAAT protocol")

        size = generator.size
        batch_size = generator.batch_size
        if size is not None:
            nb_batches = int(np.ceil(size / batch_size))
        else:
            raise ValueError("Size is None.")

        logger.info("Adversarial Training OAAT")

        if (len(self._train_params["list_swa_epoch"]) != len(self._train_params["list_swa_tau"])) or (
            len(self._train_params["list_swa_epoch"]) != len(self._list_avg_models)
        ):
            raise ValueError(
                "number of elements of list_swa_epoch must be same as of  " "list_swa_tau and list_avg_models."
            )

        beta_init = self._train_params["beta"]
        best_acc_adv_test = 0
        for i_epoch in trange(nb_epochs, desc="Adversarial Training OAAT - Epochs"):

            if i_epoch >= self._train_params["awp_warmup"]:
                self._apply_wp = True

            if i_epoch == (int(3 * nb_epochs / 4)):
                self._classifier._optimizer = torch.optim.SGD(  # pylint: disable=W0212
                    self._classifier.model.parameters(),
                    lr=self._train_params["lr"],
                    momentum=self._train_params["momentum"],
                    weight_decay=self._train_params["weight_decay"],
                )

            if isinstance(self._classifier.optimizer, torch.optim.SGD):

                if self._train_params["oaat_warmup"] == 1:
                    if i_epoch < 10:
                        self.update_learning_rate(self._classifier.optimizer, i_epoch, nb_epochs, lr_schedule="linear")
                    else:
                        self.update_learning_rate(
                            self._classifier.optimizer,
                            i_epoch,
                            nb_epochs,
                            lr_schedule=self._train_params["lr_schedule"],
                        )
                else:

                    self.update_learning_rate(
                        self._classifier.optimizer, i_epoch, nb_epochs, lr_schedule=self._train_params["lr_schedule"]
                    )

            if i_epoch > (nb_epochs // 4):
                self._train_params["i_epsilon"] = i_epoch * self._train_params["epsilon"] / nb_epochs

            if self._train_params["i_epsilon"] > self._train_params["alternate_iter_eps"]:
                self._apply_lpips_pert = True

                load_swa_model_tau = self._train_params["load_swa_model_tau"]
                file_name = f"oaat_swa_tau_{load_swa_model_tau}.model"
                if self._train_params["models_path"] is None:
                    full_path_load = os.path.join(config.ART_DATA_PATH, file_name)
                else:
                    full_path_load = os.path.join(self._train_params["models_path"], file_name)
                if os.path.isfile(full_path_load):
                    self._lpips_classifier._model._model.load_state_dict(  # pylint: disable=W0212
                        torch.load(full_path_load)
                    )
                else:
                    raise ValueError("Invalid path/file for weight average model is provided for adversarial training.")

                self._lpips_classifier.model.train(mode=False)
                self._lpips_classifier.model.to(self._classifier.device)

                self._train_params["alpha"] = self._train_params["alpha"] - self._train_params["mixup_alpha"] / (
                    nb_epochs - int(3 * nb_epochs / 4) + 1
                )
                self._train_params["i_lpips_weight"] = (
                    self._train_params["lpips_weight"]
                    * (i_epoch - int(3 * nb_epochs / 4))
                    / (nb_epochs - int(3 * nb_epochs / 4))
                )
                self._train_params["beta"] = beta_init + (self._train_params["beta_final"] - beta_init) * (
                    i_epoch - int(3 * nb_epochs / 4)
                ) / (nb_epochs - int(3 * nb_epochs / 4))

            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0

            for batch_id in range(nb_batches):
                # Create batch data
                x_batch, y_batch = generator.get_batch()
                x_batch = x_batch.copy()

                _train_loss, _train_acc, _train_n = self._batch_process(i_epoch, nb_epochs, batch_id, x_batch, y_batch)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

            train_time = time.time()

            # compute accuracy
            if validation_data is not None:
                (x_test, y_test) = validation_data
                y_test = check_and_transform_label_format(y_test, nb_classes=self.classifier.nb_classes)
                # pylint: disable=W0212
                x_preprocessed_test, y_preprocessed_test = self._classifier._apply_preprocessing(
                    x_test,
                    y_test,
                    fit=True,
                )
                # pylint: enable=W0212
                output_clean = np.argmax(self.predict(x_preprocessed_test), axis=1)
                nb_correct_clean = np.sum(output_clean == np.argmax(y_preprocessed_test, axis=1))
                self._attack.set_params(
                    eps=self._train_params["epsilon"],
                    eps_step=self._train_params["epsilon"] / 4.0,
                    max_iter=self._train_params["max_iter"],
                )
                x_test_adv = self._attack.generate(x_preprocessed_test, y=y_preprocessed_test)
                # pylint: disable=W0212
                x_preprocessed_test_adv, y_preprocessed_test = self._classifier._apply_preprocessing(
                    x_test_adv,
                    y_test,
                    fit=True,
                )
                # pylint: enable=W0212
                output_adv = np.argmax(self.predict(x_preprocessed_test_adv), axis=1)
                nb_correct_adv = np.sum(output_adv == np.argmax(y_preprocessed_test, axis=1))

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
                    self._classifier.save(filename=f"oaat_epoch_{i_epoch}", path=self._train_params["models_path"])

                # save best checkpoint
                if nb_correct_adv / x_test.shape[0] > best_acc_adv_test:
                    self._classifier.save(filename="oaat_epoch_best", path=self._train_params["models_path"])
                    best_acc_adv_test = nb_correct_adv / x_test.shape[0]

            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )

            if i_epoch > self._train_params["swa_save_epoch"] - 1:
                for epoch_swa, tau_val, p_classifier in zip(
                    self._train_params["list_swa_epoch"], self._train_params["list_swa_tau"], self._list_avg_models
                ):
                    if epoch_swa <= i_epoch:
                        file_name = f"oaat_swa_tau_{tau_val}.model"
                        if self._train_params["models_path"] is None:
                            full_path = os.path.join(config.ART_DATA_PATH, file_name)
                        else:
                            full_path = os.path.join(self._train_params["models_path"], file_name)

                        folder = os.path.split(full_path)[0]
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        # pylint: disable=W0212
                        # disable pylint because access to _modules required
                        torch.save(p_classifier._model._model.state_dict(), full_path)

    def _batch_process(
        self, i_epoch: int, nb_epochs: int, batch_id: int, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Perform the operations of OAAT for a batch of data.
        See class documentation for more information on the exact procedure.

        :param i_epoch: training epoch number.
        :param nb_epochs: total training epochs.
        :param batch_id: batch_id of input data.
        :param x_batch: batch of x.
        :param y_batch: batch of y.
        :return: tuple containing batch data loss, batch data accuracy and number of samples in the batch
        """
        import torch
        from torch import nn
        import torch.nn.functional as F

        if self._classifier.optimizer is None:
            raise ValueError("Optimizer of classifier is currently None, but is required for adversarial training.")

        if self._proxy_classifier.optimizer is None:
            raise ValueError(
                "Optimizer of proxy classifier is currently None, but is required for adversarial training."
            )

        self._classifier.model.train(mode=False)

        if self._apply_lpips_pert and (batch_id % 2 != 0):
            x_batch_pert = self._attack_lpips(
                x_batch,
                y=y_batch,
                eps=self._train_params["i_epsilon"],
                eps_step=self._train_params["i_epsilon"] / 4.0,
                max_iter=self._train_params["max_iter"],
                training_mode=False,
            )

        elif self._apply_lpips_pert and (batch_id % 2 == 0):
            self._attack.set_params(
                eps=self._train_params["mixup_epsilon"],
                eps_step=self._train_params["mixup_epsilon"] / 4.0,
                max_iter=self._train_params["max_iter"],
            )
            x_batch_pert = self._attack.generate(x_batch, y=y_batch)

        else:
            if i_epoch <= (nb_epochs // 4):
                self._attack.set_params(
                    eps=self._train_params["i_epsilon"], eps_step=self._train_params["i_epsilon"] / 2.0, max_iter=5
                )
            else:
                self._attack.set_params(
                    eps=self._train_params["i_epsilon"],
                    eps_step=self._train_params["i_epsilon"] / 4.0,
                    max_iter=self._train_params["max_iter"],
                )

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

        i_batch = torch.from_numpy(x_preprocessed).to(self._classifier.device)
        i_batch_pert = torch.from_numpy(x_preprocessed_pert).to(self._classifier.device)
        o_batch = torch.from_numpy(y_preprocessed).to(self._classifier.device)

        self._classifier.model.train(mode=True)

        if self._apply_wp:
            if not self._apply_lpips_pert:
                i_batch_pert_awp = torch.clamp(i_batch + 2.0 * (i_batch_pert - i_batch), 0, 1)
            else:
                i_batch_pert_awp = i_batch_pert
            w_perturb = self._weight_perturbation(x_batch=i_batch, x_batch_pert=i_batch_pert_awp, y_batch=o_batch)
            list_keys = list(w_perturb.keys())
            self._modify_classifier(self._classifier, list_keys, w_perturb, op="add")

        # Zero the parameter gradients
        self._classifier.optimizer.zero_grad()

        n = x_batch.shape[0]
        # Perform prediction
        model_outputs = self._classifier.model(i_batch)
        model_outputs_pert = self._classifier.model(i_batch_pert)

        # Form the loss function
        loss_clean = self._classifier.loss(model_outputs, o_batch)

        if self._apply_lpips_pert and (batch_id % 2 == 0):
            i_batch_pert_eps = torch.min(
                torch.max(i_batch_pert, i_batch - self._train_params["i_epsilon"]),
                i_batch + self._train_params["i_epsilon"],
            )
            model_outputs_pert_eps = self._classifier.model(i_batch_pert_eps)
            loss_kl = (1.0 / n) * nn.KLDivLoss(reduction="sum")(
                F.log_softmax(model_outputs_pert_eps, dim=1),
                torch.clamp(
                    self._train_params["alpha"] * F.softmax(model_outputs, dim=1)
                    + (1.0 - self._train_params["alpha"]) * F.softmax(model_outputs_pert, dim=1),
                    min=EPS,
                ),
            )

        else:
            loss_kl = (1.0 / n) * nn.KLDivLoss(reduction="sum")(
                F.log_softmax(model_outputs_pert, dim=1), torch.clamp(F.softmax(model_outputs, dim=1), min=EPS)
            )
        loss = loss_clean + self._train_params["beta"] * loss_kl

        loss.backward()

        self._classifier.optimizer.step()

        if self._apply_wp:
            self._modify_classifier(self._classifier, list_keys, w_perturb, op="subtract")

        train_loss = loss.item() * o_batch.size(0)
        train_acc = (model_outputs_pert.max(1)[1] == o_batch).sum().item()
        train_n = o_batch.size(0)

        for epoch_swa, tau_val, p_classifier in zip(
            self._train_params["list_swa_epoch"], self._train_params["list_swa_tau"], self._list_avg_models
        ):
            if i_epoch == epoch_swa:
                for name, params in self._classifier.model.state_dict().items():
                    p_classifier.model.state_dict()[name] = params
            elif i_epoch > epoch_swa:
                for name, params in self._classifier.model.state_dict().items():
                    p_classifier.model.state_dict()[name] = (
                        1 - tau_val
                    ) * params + tau_val * p_classifier.model.state_dict()[name]
            else:
                pass

        self._classifier.model.train(mode=False)

        return train_loss, train_acc, train_n

    def _weight_perturbation(
        self, x_batch: "torch.Tensor", x_batch_pert: "torch.Tensor", y_batch: "torch.Tensor"
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
        self._proxy_classifier.model.load_state_dict(self._classifier.model.state_dict())
        self._proxy_classifier.model.train(mode=True)

        n = x_batch.shape[0]
        # Perform prediction
        model_outputs = self._proxy_classifier.model(x_batch)
        model_outputs_pert = self._proxy_classifier.model(x_batch_pert)
        loss_clean = self._proxy_classifier.loss(model_outputs, y_batch)
        loss_kl = (1.0 / n) * nn.KLDivLoss(reduction="sum")(
            F.log_softmax(model_outputs_pert, dim=1), torch.clamp(F.softmax(model_outputs, dim=1), min=EPS)
        )
        loss = -1.0 * (loss_clean + self._train_params["beta"] * loss_kl)

        self._proxy_classifier.optimizer.zero_grad()
        loss.backward()
        self._proxy_classifier.optimizer.step()

        params_dict_proxy, _ = self._calculate_model_params(self._proxy_classifier)

        for name in list_keys:
            perturbation = params_dict_proxy[name]["param"] - params_dict[name]["param"]
            perturbation = torch.reshape(perturbation, list(params_dict[name]["size"]))
            scale = params_dict[name]["norm"] / (perturbation.norm() + EPS)
            w_perturb[name] = scale * perturbation

        return w_perturb

    @staticmethod
    def _calculate_model_params(
        p_classifier: PyTorchClassifier,
    ) -> Tuple[Dict[str, Dict[str, "torch.Tensor"]], "torch.Tensor"]:
        """
        Calculates a given model's different layers' parameters' shape and norm, and model parameter norm.

        :param p_classifier: model for awp protocol.
        :return: tuple with first element a dictionary with model parameters' names as keys and a nested dictionary
        as value. The nested dictionary contains model parameters, model parameters' size, model parameters' norms.
        The second element of tuple denotes norm of all model parameters
        """
        import torch

        params_dict: Dict[str, Dict[str, "torch.Tensor"]] = OrderedDict()
        list_params = []
        for name, param in p_classifier.model.state_dict().items():
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
        Modify the model's weight parameters according to the weight perturbations.

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
            for name, param in p_classifier.model.named_parameters():
                if name in list_keys:
                    param.add_(c_mult * self._train_params["awp_gamma"] * w_perturb[name])

    @staticmethod
    def get_layer_activations(  # type: ignore
        p_classifier: PyTorchClassifier,
        x: "torch.Tensor",
        layers: List[Union[int, str]],
    ) -> Tuple[Dict[str, "torch.Tensor"], List[str]]:
        """
        Return the output of the specified layers for input `x`. `layers` is a list of either layer indices (between 0
        and `nb_layers - 1`) or layers' name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param p_classifier: model for adversarial training protocol.
        :param x: Input for computing the activations.
        :param layers: Layers for computing the activations
        :return: Tuple containing the output dict and a list of layers' names. In dictionary each element is a
                 layer's output where the first dimension is the batch size corresponding to `x'.
        """

        p_classifier.model.train(mode=False)

        list_layer_names = []
        for layer in layers:
            if isinstance(layer, six.string_types):
                if layer not in p_classifier._layer_names:  # pylint: disable=W0212
                    raise ValueError(f"Layer name {layer} not supported")
                layer_name = layer
                list_layer_names.append(layer_name)

            elif isinstance(layer, int):
                layer_index = layer
                layer_name = p_classifier._layer_names[layer_index]  # pylint: disable=W0212
                list_layer_names.append(layer_name)

            else:
                raise TypeError("Layer must be of type str or int")

        def get_feature(name):
            # the hook signature
            def hook(model, input, output):  # pylint: disable=W0622,W0613
                p_classifier._features[name] = output  # pylint: disable=W0212

            return hook

        if not hasattr(p_classifier, "_features"):
            p_classifier._features = {}  # pylint: disable=W0212
            # register forward hooks on the layers of choice

        for layer_name in list_layer_names:
            if layer_name not in p_classifier._features:  # pylint: disable=W0212
                interim_layer = dict([*p_classifier._model._model.named_modules()])[layer_name]  # pylint: disable=W0212
                interim_layer.register_forward_hook(get_feature(layer_name))

        p_classifier.model(x)
        return p_classifier._features, list_layer_names  # pylint: disable=W0212

    @staticmethod
    def normalize_concatenate_activations(
        activations_dict: Dict[str, "torch.Tensor"],
        list_layer_names: List[str],
    ) -> "torch.Tensor":
        """
        Takes a dictionary `activations_dict' of activation values of different layers for an input batch and Returns
        a tensor where all activation values are normalised layer-wise and flattened to a vector for
        each input of the batch.

        :param activations_dict: dict containing the activations at different layers.
        :param list_layer_names: Layers' names for fetching the activations
        :return: The activations after normalisation and flattening, where the first dimension is the batch size.
        """

        import torch

        activation_vectors = []
        for name in list_layer_names:
            temp_activation = activations_dict[name]
            size_temp_activation = list(temp_activation.size())
            norm_factor_layer = size_temp_activation[2] * size_temp_activation[3]
            norm_temp_activation = torch.sqrt(torch.sum(temp_activation ** 2, dim=1, keepdim=True)) + EPS
            temp_activation_n_channel = temp_activation / norm_temp_activation
            temp_activation_n_layer_channel = temp_activation_n_channel / np.sqrt(norm_factor_layer)
            temp_activation_n_layer_channel_flat = temp_activation_n_layer_channel.view(size_temp_activation[0], -1)
            activation_vectors.append(temp_activation_n_layer_channel_flat)

        activation_vectors_flattened = torch.cat(activation_vectors, dim=1)
        return activation_vectors_flattened

    @staticmethod
    def calculate_lpips_distance(  # type: ignore
        p_classifier: PyTorchClassifier,
        input_1: "torch.Tensor",
        input_2: "torch.Tensor",
        layers: List[Union[int, str]],
    ) -> "torch.Tensor":
        """
        Return the LPIPS distance between input_1 and input_2. `layers` is a list of either layer indices (between 0 and
        `nb_layers - 1`) or layers' name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param p_classifier: model for adversarial training protocol.
        :param input_1: Input for computing the activations.
        :param input_2: Input for computing the activations.
        :param layers: Layers for computing the activations.
        :return: The lpips distance, where the first dimension is the batch size corresponding to
                `input_1`.
        """

        import torch

        activations_input_1, list_layer_names = AdversarialTrainerOAATPyTorch.get_layer_activations(
            p_classifier, input_1, layers
        )
        activations_input_2, _ = AdversarialTrainerOAATPyTorch.get_layer_activations(p_classifier, input_2, layers)
        normalised_activations_1 = AdversarialTrainerOAATPyTorch.normalize_concatenate_activations(
            activations_input_1, list_layer_names
        )
        normalised_activations_2 = AdversarialTrainerOAATPyTorch.normalize_concatenate_activations(
            activations_input_2, list_layer_names
        )
        lpips_distance = torch.norm(normalised_activations_1 - normalised_activations_2, dim=1)

        return lpips_distance

    def update_learning_rate(
        self, optimizer: "torch.optim.optimizer.Optimizer", epoch: int, nb_epochs: int, lr_schedule: str = "step"
    ) -> None:
        """
        adjust learning rate of the optimizer.

        :param optimizer: optimizer of the classifier.
        :param epoch: current training epoch.
        :param nb_epochs: total training epoch.
        :param lr_schedule: string denoting learning rate scheduler for optimizer
        :return: calculated learning rate
        """
        if lr_schedule.lower() == "cosine":
            if self._train_params["oaat_warmup"] == 1:
                l_r = self._train_params["lr"] * 0.5 * (1 + np.cos((epoch - 10) / (nb_epochs - 10) * np.pi))
            else:
                l_r = self._train_params["lr"] * 0.5 * (1 + np.cos(epoch / nb_epochs * np.pi))

            for param_group in optimizer.param_groups:
                param_group["lr"] = l_r

        elif lr_schedule.lower() == "linear":
            l_r = (epoch + 1) * (self._train_params["lr"] / 10)

            for param_group in optimizer.param_groups:
                param_group["lr"] = l_r

        elif lr_schedule.lower() == "step":
            if epoch >= 75 * nb_epochs / 110:
                l_r = self._train_params["lr"] * 0.1
            if epoch >= 90 * nb_epochs / 110:
                l_r = self._train_params["lr"] * 0.01
            if epoch >= 100 * nb_epochs / 110:
                l_r = self._train_params["lr"] * 0.001

            for param_group in optimizer.param_groups:
                param_group["lr"] = l_r
        else:
            raise ValueError(f"lr_schedule {lr_schedule} not supported")

    def _attack_lpips(
        self,
        x: np.ndarray,
        y: np.ndarray,
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        max_iter: int,
        training_mode: bool,
    ) -> np.ndarray:
        """
        Compute adversarial examples with cross entropy and lpips distance.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236).
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: number of iterations of attack.
        :param training_mode: training mode of classifier for attack generation. training_mode=False puts classifier in
                                eval mode
        :return: Adversarial examples.
        """
        import torch

        x_t = torch.from_numpy(x.astype(ART_NUMPY_DTYPE)).to(self._classifier.device)
        y_t = torch.from_numpy(y.astype(ART_NUMPY_DTYPE)).to(self._classifier.device)
        adv_x = torch.clone(x_t)

        for i_max_iter in range(max_iter):
            adv_x = self._one_step_adv_example(adv_x, x_t, y_t, eps, eps_step, i_max_iter == 0, training_mode)

        return adv_x.cpu().detach().numpy()

    def _one_step_adv_example(
        self,
        x: "torch.Tensor",
        x_init: "torch.Tensor",
        y: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        random_init: bool,
        training_mode: bool,
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236).
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :param training_mode: training mode of classifier for attack generation. training_mode=False puts classifier in
                                eval mode
        :return: Adversarial examples.
        """
        import torch

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation_array = (
                random_sphere(n, m, eps, self._train_params["norm"]).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            )
            random_perturbation = torch.from_numpy(random_perturbation_array).to(self._classifier.device)
            x_adv = x + random_perturbation

            if self._classifier.clip_values is not None:
                clip_min, clip_max = self._classifier.clip_values
                x_adv = torch.max(
                    torch.min(x_adv, torch.tensor(clip_max).to(self._classifier.device)),
                    torch.tensor(clip_min).to(self._classifier.device),
                )

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation(x_adv, x_init, y, training_mode)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self._train_params["norm"])

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    def _compute_perturbation(
        self, x: "torch.Tensor", x_init: "torch.Tensor", y: "torch.Tensor", training_mode: bool = False
    ) -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param x_init: initial input samples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param training_mode: training mode of classifier for attack generation. training_mode=False puts classifier in
                                eval mode
        """
        import torch

        self._classifier.model.train(mode=training_mode)
        self._lpips_classifier.model.train(mode=training_mode)

        # Backpropagation through RNN modules in eval mode raises RuntimeError due to cudnn issues and require training
        # mode, i.e. RuntimeError: cudnn RNN backward can only be called in training mode. Therefore, if the model is
        # an RNN type we always use training mode but freeze batch-norm and dropout layers if training_mode=False.
        if self._classifier.is_rnn:
            self._classifier.model.train(mode=True)
            if not training_mode:
                logger.debug(
                    "Freezing batch-norm and dropout layers for gradient calculation in train mode with eval parameters"
                    "of batch-norm and dropout."
                )
                self._classifier.set_batchnorm(train=False)
                self._classifier.set_dropout(train=False)

        if self._lpips_classifier.is_rnn:
            self._lpips_classifier.model.train(mode=True)
            if not training_mode:
                logger.debug(
                    "Freezing batch-norm and dropout layers for gradient calculation in train mode with eval parameters"
                    "of batch-norm and dropout."
                )
                self._lpips_classifier.set_batchnorm(train=False)
                self._lpips_classifier.set_dropout(train=False)

        # Apply preprocessing
        if self._classifier.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                x_grad = x.clone().detach().requires_grad_(True)
            else:
                x_grad = torch.tensor(x).to(self._classifier.device)
                x_grad.requires_grad = True
            if isinstance(y, torch.Tensor):
                y_grad = y.clone().detach()
            else:
                y_grad = torch.tensor(y).to(self._classifier.device)
            if isinstance(x_init, torch.Tensor):
                x_init = x_init.clone().detach()
            else:
                x_init = torch.tensor(x_init).to(self._classifier.device)
            inputs_t, y_preprocessed = self._classifier._apply_preprocessing(  # pylint: disable=W0212
                x_grad, y=y_grad, fit=False, no_grad=False
            )
            inputs_init, _ = self._classifier._apply_preprocessing(  # pylint: disable=W0212
                x_init, y=y_grad, fit=False, no_grad=False
            )
        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._classifier._apply_preprocessing(  # pylint: disable=W0212
                x, y=y, fit=False, no_grad=True
            )
            x_grad = torch.from_numpy(x_preprocessed).to(self._classifier.device)
            x_grad.requires_grad = True
            inputs_t = x_grad
            x_init_preprocessed, _ = self._classifier._apply_preprocessing(  # pylint: disable=W0212
                x_init, y=y, fit=False, no_grad=True
            )
            x_init = torch.from_numpy(x_init_preprocessed).to(self._classifier.device)
            x_init.requires_grad = False
            inputs_init = x_init
        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        # Check label shape
        if self._classifier._reduce_labels:  # pylint: disable=W0212
            y_preprocessed = self._classifier.reduce_labels(y_preprocessed)

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._classifier.device)
        else:
            labels_t = y_preprocessed

        # Compute the gradient and return
        model_outputs = self._classifier.model(inputs_t)

        lpips_distance = self.calculate_lpips_distance(
            p_classifier=self._lpips_classifier,
            input_1=inputs_t,
            input_2=inputs_init,
            layers=self._train_params["layer_names_activation"],
        )
        loss = self._classifier.loss(model_outputs, labels_t) - self._train_params["i_lpips_weight"] * torch.mean(
            lpips_distance
        )

        # Clean gradients
        self._classifier.model.zero_grad()
        self._lpips_classifier.model.zero_grad()

        loss.backward()

        if x_grad.grad is not None:
            grad = x_grad.grad
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")

        if not self._classifier.all_framework_preprocessing:
            grad = self._classifier._apply_preprocessing_gradient(x, grad)  # pylint: disable=W0212
        assert grad.shape == x.shape

        # Check for nan before normalisation and replace with 0
        if torch.any(grad.isnan()):  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad[grad.isnan()] = 0.0

        # Apply norm bound
        if self._train_params["norm"] in ["inf", np.inf]:
            grad = grad.sign()

        elif self._train_params["norm"] == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + EPS)  # type: ignore

        elif self._train_params["norm"] == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + EPS)  # type: ignore

        assert x.shape == grad.shape

        return grad

    def _apply_perturbation(
        self, x: "torch.Tensor", perturbation: "torch.Tensor", eps_step: Union[int, float, np.ndarray]
    ) -> "torch.Tensor":
        """
        Apply perturbation on examples.

        :param x: Current adversarial examples.
        :param perturbation: Current perturbations.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch

        eps_step = np.array(eps_step, dtype=ART_NUMPY_DTYPE)
        perturbation_step = torch.tensor(eps_step).to(self._classifier.device) * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        x = x + perturbation_step
        if self._classifier.clip_values is not None:
            clip_min, clip_max = self._classifier.clip_values
            x = torch.max(
                torch.min(x, torch.tensor(clip_max).to(self._classifier.device)),
                torch.tensor(clip_min).to(self._classifier.device),
            )

        return x

    def _projection(
        self, values: "torch.Tensor", eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]
    ) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: Maximum norm allowed.
        :param norm_p: L_p norm to use for clipping supporting 1, 2, `np.Inf` and "inf".
        :return: Values of `values` after projection.
        """
        import torch

        values_tmp = values.reshape(values.shape[0], -1)

        if norm_p == 2:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 2."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(self._classifier.device),
                    eps / (torch.norm(values_tmp, p=2, dim=1) + EPS),
                ).unsqueeze_(-1)
            )

        elif norm_p == 1:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 1."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(self._classifier.device),
                    eps / (torch.norm(values_tmp, p=1, dim=1) + EPS),
                ).unsqueeze_(-1)
            )

        elif norm_p in [np.inf, "inf"]:
            if isinstance(eps, np.ndarray):
                eps_array = eps * np.ones_like(values.cpu())
                eps = eps_array.reshape([eps_array.shape[0], -1])

            values_tmp = values_tmp.sign() * torch.min(
                values_tmp.abs(), torch.tensor([eps], dtype=torch.float32).to(self._classifier.device)
            )

        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported."
            )

        values = values_tmp.reshape(values.shape)

        return values  # pylint: disable=C0302
