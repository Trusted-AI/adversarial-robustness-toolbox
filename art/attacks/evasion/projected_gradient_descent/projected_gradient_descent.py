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
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on an lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class ProjectedGradientDescent(EvasionAttack):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "max_iter",
        "random_eps",
        "tensor_board",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        tensor_board: Union[str, bool] = False,
        verbose: bool = True,
    ):
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this
                           method with PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param tensor_board: Activate summary writer for TensorBoard: Default is `False` and deactivated summary writer.
                             If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory. Provide `path` in type
                             `str` to save in path/CURRENT_DATETIME_HOSTNAME.
                             Use hierarchical folder structure to compare between runs easily. e.g. pass in ‘runs/exp1’,
                             ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator, tensor_board=tensor_board)

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.random_eps = random_eps
        self.verbose = verbose
        ProjectedGradientDescent._check_params(self)

        self._attack: Union[
            ProjectedGradientDescentPyTorch, ProjectedGradientDescentTensorFlowV2, ProjectedGradientDescentNumpy
        ]
        if isinstance(self.estimator, PyTorchClassifier) and self.estimator.all_framework_preprocessing:
            self._attack = ProjectedGradientDescentPyTorch(
                estimator=estimator,  # type: ignore
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                num_random_init=num_random_init,
                batch_size=batch_size,
                random_eps=random_eps,
                tensor_board=tensor_board,
                verbose=verbose,
            )

        elif isinstance(self.estimator, TensorFlowV2Classifier) and self.estimator.all_framework_preprocessing:
            self._attack = ProjectedGradientDescentTensorFlowV2(
                estimator=estimator,  # type: ignore
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                num_random_init=num_random_init,
                batch_size=batch_size,
                random_eps=random_eps,
                verbose=verbose,
            )

        else:
            self._attack = ProjectedGradientDescentNumpy(
                estimator=estimator,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                num_random_init=num_random_init,
                batch_size=batch_size,
                random_eps=random_eps,
                tensor_board=tensor_board,
                verbose=verbose,
            )

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        logger.info("Creating adversarial samples.")
        return self._attack.generate(x=x, y=y, **kwargs)

    def set_params(self, **kwargs) -> None:
        super().set_params(**kwargs)
        self._attack.set_params(**kwargs)

    def _check_params(self) -> None:

        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')

        if not (
            isinstance(self.eps, (int, float))
            and isinstance(self.eps_step, (int, float))
            or isinstance(self.eps, np.ndarray)
            and isinstance(self.eps_step, np.ndarray)
        ):
            raise TypeError(
                "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same type of `int`"
                ", `float`, or `np.ndarray`."
            )

        if isinstance(self.eps, (int, float)):
            if self.eps < 0:
                raise ValueError("The perturbation size `eps` has to be nonnegative.")
        else:
            if (self.eps < 0).any():
                raise ValueError("The perturbation size `eps` has to be nonnegative.")

        if isinstance(self.eps_step, (int, float)):
            if self.eps_step <= 0:
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")
        else:
            if (self.eps_step <= 0).any():
                raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
            if self.eps.shape != self.eps_step.shape:
                raise ValueError(
                    "The perturbation size `eps` and the perturbation step-size `eps_step` must have the same shape."
                )

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, (int, np.int)):
            raise TypeError("The number of random initialisations has to be of type integer.")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if self.max_iter < 0:
            raise ValueError("The number of iterations `max_iter` has to be a nonnegative integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The verbose has to be a Boolean.")

        if not isinstance(self.tensor_board, (bool, str)):
            raise ValueError("The argument `tensor_board` has to be either of type bool or str.")
