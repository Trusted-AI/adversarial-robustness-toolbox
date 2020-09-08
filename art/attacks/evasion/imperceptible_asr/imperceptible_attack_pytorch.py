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
This module implements the imperceptible, robust, and targeted attack to generate adversarial examples for automatic
speech recognition models. This attack will be implemented specifically for DeepSpeech model and is framework dependent,
specifically for Pytorch.

| Paper link: https://arxiv.org/abs/1903.10346
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech

if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class ImperceptibleAttackPytorch(EvasionAttack):
    """
    This class implements the imperceptible, robust, and targeted attack to generate adversarial examples for automatic
    speech recognition models. This attack will be implemented specifically for DeepSpeech model and is framework
    dependent, specifically for Pytorch.

    | Paper link: https://arxiv.org/abs/1903.10346
    """

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "max_iter",
        "batch_size",
    ]

    _estimator_requirements = (
        BaseEstimator,
        LossGradientsMixin,
        NeuralNetworkMixin,
        SpeechRecognizerMixin,
        PyTorchEstimator,
        PyTorchDeepSpeech,
    )

    def __init__(
        self,
        estimator: PyTorchDeepSpeech,
        initial_eps: float = 2000,
        max_iter_1st_stage: int = 1000,
        max_iter_2nd_stage: int = 4000,
        learning_rate_1st_stage: float = 0.1,
        learning_rate_2nd_stage: float = 0.001,
        optimizer_1st_stage: "Optimizer" = torch.optim.SGD,
        optimizer_2nd_stage: "Optimizer" = torch.optim.SGD,
        batch_size: int = 32,
    ):
        """
        Create a :class:`.ImperceptibleAttackPytorch` instance.

        :param estimator: A trained estimator.
        :param initial_eps: Initial maximum perturbation that the attacker can introduce.
        :param max_iter_1st_stage: The maximum number of iterations applied for the first stage of the optimization of
                                   the attack.
        :param max_iter_2nd_stage: The maximum number of iterations applied for the second stage of the optimization of
                                   the attack.
        :param learning_rate_1st_stage: The initial learning rate applied for the first stage of the optimization of
                                        the attack.
        :param learning_rate_2nd_stage: The initial learning rate applied for the second stage of the optimization of
                                        the attack.
        :param optimizer_1st_stage: The optimizer applied for the first stage of the optimization of the attack.
        :param optimizer_2nd_stage: The optimizer applied for the second stage of the optimization of the attack.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        if (
            hasattr(estimator, "preprocessing")
            and (estimator.preprocessing is not None and estimator.preprocessing != (0, 1))
        ) or (
            hasattr(estimator, "preprocessing_defences")
            and (estimator.preprocessing_defences is not None and estimator.preprocessing_defences != [])
        ):
            raise NotImplementedError(
                "The framework-specific implementation currently does not apply preprocessing and "
                "preprocessing defences."
            )

        super(ImperceptibleAttackPytorch, self).__init__(estimator=estimator)

        self.initial_eps = initial_eps
        self.max_iter_1st_stage = max_iter_1st_stage
        self.max_iter_2nd_stage = max_iter_2nd_stage
        self.learning_rate_1st_stage = learning_rate_1st_stage
        self.learning_rate_2nd_stage = learning_rate_2nd_stage
        self.optimizer_1st_stage = optimizer_1st_stage
        self.optimizer_2nd_stage = optimizer_2nd_stage
        self.batch_size = batch_size

    def generate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`. Note that, this
                  only supports targeted attack.
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # Get the mask
        mask = self._get_mask(x, **kwargs)

        # Create dataset
        if mask is not None:
            # Here we need to make a distinction: if the masks are different for each input, we need to index
            # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
            if len(mask.shape) == len(x.shape):
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)),
                )

            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])),
                )

        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(ART_NUMPY_DTYPE)), torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples
        adv_x_best = None
        rate_best = None

            # Compute perturbation with batching
            for (batch_id, batch_all) in enumerate(data_loader):
                if mask is not None:
                    (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
                else:
                    (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                adv_x[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels, mask_batch)


        return adv_x_best

    def _generate_batch(self, x: "torch.Tensor", targets: "torch.Tensor", mask: "torch.Tensor") -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :return: Adversarial examples.
        """
        inputs = x.to(self.estimator.device)
        targets = targets.to(self.estimator.device)
        adv_x = inputs

        if mask is not None:
            mask = mask.to(self.estimator.device)

        for i_max_iter in range(self.max_iter):
            adv_x = self._compute_torch(
                adv_x, inputs, targets, mask, self.eps, self.eps_step, self.num_random_init > 0 and i_max_iter == 0,
            )

        return adv_x.cpu().detach().numpy()

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if self.initial_eps <= 0:
            raise ValueError("The perturbation size `initial_eps` has to be positive.")

        if not isinstance(self.max_iter_1st_stage, int):
            raise ValueError("The maximum number of iterations must be of type int.")
        if not self.max_iter_1st_stage > 0:
            raise ValueError("The maximum number of iterations must be greater than 0.")

        if not isinstance(self.max_iter_2nd_stage, int):
            raise ValueError("The maximum number of iterations must be of type int.")
        if not self.max_iter_2nd_stage > 0:
            raise ValueError("The maximum number of iterations must be greater than 0.")

        if not isinstance(self.learning_rate_1st_stage, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate_1st_stage > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.learning_rate_2nd_stage, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate_2nd_stage > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")
