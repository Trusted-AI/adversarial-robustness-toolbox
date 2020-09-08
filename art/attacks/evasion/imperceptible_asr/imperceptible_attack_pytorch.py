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
        global_max_length: int = 10000,
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
        :param global_max_length: The length of the longest audio signal allowed by this attack.
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

        # Set attack attributes
        self.initial_eps = initial_eps
        self.max_iter_1st_stage = max_iter_1st_stage
        self.max_iter_2nd_stage = max_iter_2nd_stage
        self.learning_rate_1st_stage = learning_rate_1st_stage
        self.learning_rate_2nd_stage = learning_rate_2nd_stage
        self.optimizer_1st_stage = optimizer_1st_stage
        self.optimizer_2nd_stage = optimizer_2nd_stage
        self.global_max_length = global_max_length
        self.batch_size = batch_size

        # Check validity of attack attributes
        self._check_params()

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


        # Start to compute adversarial examples
        # adv_x_best = None
        # rate_best = None
        #
        #     # Compute perturbation with batching
        #     for (batch_id, batch_all) in enumerate(data_loader):
        #         if mask is not None:
        #             (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
        #         else:
        #             (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None
        #
        #         batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
        #         adv_x[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels, mask_batch)


        return

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
        return

    def _partial_forward(self, batch_size: int):
        """

        :param global_max_length:
        :return:
        """
        from torch.autograd import Variable

        self.global_delta = Variable(
            x=torch.zeros(self.batch_size, self.global_max_length).type(torch.FloatTensor),
            requires_grad=True
        )
        self.global_delta.to(self.estimator.device)


    def _attack_1st_stage(self):
        return

    def _attack_2nd_stage(self):
        return

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

        if not isinstance(self.global_max_length, int):
            raise ValueError("The length of the longest audio signal must be of type int.")
        if not self.global_max_length > 0:
            raise ValueError("The length of the longest audio signal must be greater than 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")
