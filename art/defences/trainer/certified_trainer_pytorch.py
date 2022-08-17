# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements certified adversarial training following ______.

| Paper link:

"""
import logging
from typing import Optional, Union, Any, TYPE_CHECKING
import random

import numpy as np
import torch

from tqdm import tqdm

from art.defences.trainer.trainer import Trainer
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.utils import check_and_transform_label_format
from art.estimators.certification.deep_z import PytorchDeepZ


if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class DefaultLinearScheduler:
    def __init__(self, step_per_epoch, bound=0.0):
        self.step_per_epoch = step_per_epoch
        self.bound = bound

    def step(self):
        self.bound += self.step_per_epoch
        return self.bound


class AdversarialTrainerCertified(Trainer):
    """
    Class performing adversarial training...

    |

    """

    def __init__(
        self,
        classifier: PytorchDeepZ,
        nb_epochs: Optional[int] = 205,
        batch_size: Optional[int] = 128,
        eps: Union[int, float] = 8,
        eps_step: Union[int, float] = 2,
        max_iter: int = 7,
        num_random_init: int = 1,
    ) -> None:
        """
        Create an :class:`.AdversarialTrainerCertified` instance.

        Default values are for CIFAR-10 in pixel range 0-255.

        :param classifier: Classifier to train adversarially.
        :param nb_epochs: Number of training epochs.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
                                starting at the original input.
        """
        super().__init__(classifier=classifier)  # type: ignore
        self._classifier: PytorchDeepZ
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        # Setting up adversary and perform adversarial training:
        self.attack = ProjectedGradientDescent(
            classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            num_random_init=num_random_init,
        )

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        certification_loss: str = "interval_loss_cce",
        batch_size: int = 10,
        nb_epochs: int = 10,
        training_mode: bool = True,
        scheduler: Optional[Any] = None,
        bound: float = 0.1,
        loss_weighting: float = 0.1,
        use_certification_schedule: bool = True,
        certification_schedule: Optional[Any] = None,
        pgd_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param initial_zonotope_abstraction_bounds:
        :param certification_loss: which certification loss function to use. Currently, by default
                                   supports "interval_loss_cce" or "max_logit_loss".
        :param batch_size: Size of batches to use for certified training. NB, this will run the data
                           sequentially accumulating gradients over the batch size.
        :param loss_weighting:
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param certification_schedule: Schedule for gradually increasing the certification radius. Empirical studies
                                       have shown that this is often required to achieve best performance.
                                       Either True to use the default linear scheduler,
                                       or a class with a .step() method that returns the updated bound every epoch.

        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """

        # Set model mode
        self._classifier._model.train(mode=training_mode)  # pylint: disable=W0212
        if pgd_params is None:
            pgd_params = {"eps": 0.25,
                          "eps_step": 0.05,
                          "max_iter": 20,
                          "batch_size": 128,
                          "num_random_init": 1}

        if self._classifier._optimizer is None:  # pragma: no cover # pylint: disable=W0212
            raise ValueError("An optimizer is needed to train the model, but none is provided.")

        y = check_and_transform_label_format(y, nb_classes=self._classifier.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._classifier.apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self._classifier.reduce_labels(y_preprocessed)

        num_batch = int(np.ceil(len(x_preprocessed) / float(pgd_params["batch_size"])))
        ind = np.arange(len(x_preprocessed))
        from sklearn.utils import shuffle

        x_cert = np.copy(x_preprocessed)
        y_cert = np.copy(y_preprocessed)

        # Start training
        if use_certification_schedule:
            print(certification_schedule)
            if certification_schedule is None:
                certification_schedule_function = DefaultLinearScheduler(step_per_epoch=bound / nb_epochs,
                                                                         bound=0.0)

        for epoch in tqdm(range(nb_epochs)):
            if use_certification_schedule:
                bound = certification_schedule_function.step()
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                certified_loss = torch.tensor(0.0).to(self._classifier.device)
                samples_certified = 0
                # Zero the parameter gradients
                self._classifier._optimizer.zero_grad()  # pylint: disable=W0212

                # get the certified loss
                x_cert, y_cert = shuffle(x_cert, y_cert)
                for i, (sample, label) in enumerate(zip(x_cert, y_cert)):
                    eps_bound = np.eye(784) * bound  # TODO Generalise this line
                    self._classifier.set_forward_mode("concrete")
                    concrete_pred = self._classifier.model.forward(sample)
                    concrete_pred = torch.argmax(concrete_pred)
                    processed_sample, eps_bound = self._classifier.pre_process(cent=np.copy(sample), eps=eps_bound)
                    processed_sample = np.expand_dims(processed_sample, axis=0)

                    # Perform prediction
                    self._classifier.set_forward_mode("abstract")
                    bias, eps = self._classifier.model.forward(eps=eps_bound, cent=processed_sample)
                    # Form the loss function
                    bias = torch.unsqueeze(bias, dim=0)

                    if certification_loss == "max_logit_loss":
                        certified_loss += self._classifier.max_logit_loss(
                            output=torch.cat((bias, eps)), target=np.expand_dims(label, axis=0)
                        )
                    elif certification_loss == "interval_loss_cce":
                        certified_loss += self._classifier.interval_loss_cce(
                            prediction=torch.cat((bias, eps)),
                            target=torch.from_numpy(np.expand_dims(label, axis=0)).to(self._classifier.device)
                        )
                    certification_results = []
                    bias = torch.squeeze(bias).detach().cpu().numpy()
                    eps = eps.detach().cpu().numpy()

                    for k in range(self._classifier.nb_classes):
                        if k != concrete_pred:
                            cert_via_sub = self._classifier.certify_via_subtraction(
                                predicted_class=concrete_pred, class_to_consider=k, cent=bias, eps=eps
                            )
                            certification_results.append(cert_via_sub)

                    if all(certification_results):
                        samples_certified += 1

                    if (i + 1) % batch_size == 0 and i > 0:
                        break

                certified_loss /= batch_size
                # Concrete PGD loss
                i_batch = np.copy(x_preprocessed[ind[m * pgd_params["batch_size"] : (m + 1) * pgd_params["batch_size"]]]).astype("float32")
                o_batch = y_preprocessed[ind[m * pgd_params["batch_size"] : (m + 1) * pgd_params["batch_size"]]]

                # Perform prediction
                self._classifier.set_forward_mode("concrete")
                self.attack = ProjectedGradientDescent(estimator=self._classifier,
                                                       eps=pgd_params["eps"],
                                                       eps_step=pgd_params["eps_step"],
                                                       max_iter=pgd_params["max_iter"],
                                                       num_random_init=pgd_params["num_random_init"],
                )
                i_batch = self.attack.generate(i_batch, y=o_batch)
                self._classifier.model.zero_grad()
                model_outputs = self._classifier.model.forward(i_batch)
                acc = self._classifier.get_accuracy(model_outputs, o_batch)

                # Form the loss function
                pgd_loss = self._classifier.concrete_loss(
                    model_outputs, torch.from_numpy(o_batch).to(self._classifier.device)
                )  # pylint: disable=W0212
                print("")
                print(f"Epoch {epoch}, Batch {m}/{num_batch}:")
                print(f"Loss is {pgd_loss} Cert Loss is {certified_loss}")
                print(f"Acc is {acc} Cert Acc is {samples_certified / batch_size}")
                loss = certified_loss * loss_weighting + pgd_loss * (1 - loss_weighting)
                # Do training
                if self._classifier._use_amp:  # pragma: no cover # pylint: disable=W0212
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._classifier._optimizer) as scaled_loss:  # pylint: disable=W0212
                        scaled_loss.backward()

                else:
                    loss.backward()

                self._classifier._optimizer.step()  # pylint: disable=W0212

            if scheduler is not None:
                scheduler.step()
