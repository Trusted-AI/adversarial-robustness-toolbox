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
from typing import Optional

import numpy as np
from scipy.stats import truncnorm

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import (
    ClassifierMixin,
    ClassifierGradients,
)
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
)

logger = logging.getLogger(__name__)


class ProjectedGradientDescentCommon(FastGradientMethod):
    """
    Common class for different variations of implementation of the Projected Gradient Descent attack. The attack is an
    iterative method in which, after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted data range). This is the
    attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    attack_params = FastGradientMethod.attack_params + ["max_iter", "random_eps"]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: ClassifierGradients,
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescentCommon` instance.

        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation supporting np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
            suggests this for FGSM based training to generalize across different epsilons. eps_step is
            modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
            is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        super(ProjectedGradientDescentCommon, self).__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            minimal=False,
        )
        self.max_iter = max_iter
        self.random_eps = random_eps
        ProjectedGradientDescentCommon._check_params(self)

        if self.random_eps:
            lower, upper = 0, eps
            mu, sigma = 0, (eps / 2)
            self.norm_dist = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    def _random_eps(self):
        """
        Check whether random eps is enabled, then scale eps and eps_step appropriately.
        """
        if self.random_eps:
            ratio = self.eps_step / self.eps
            self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            self.eps_step = ratio * self.eps

    def _set_targets(self, x, y, classifier_mixin=True):
        """
        Check and set up targets.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :type classifier_mixin: `bool`
        :return: The targets.
        :rtype: `np.ndarray`
        """
        if classifier_mixin:
            y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            if classifier_mixin:
                targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets

    @staticmethod
    def _get_mask(x, classifier_mixin=True, **kwargs):
        """
        Get the mask from the kwargs.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :type classifier_mixin: `bool`
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: The mask.
        :rtype: `np.ndarray`
        """
        mask = kwargs.get("mask")

        if mask is not None:
            if classifier_mixin:
                # Ensure the mask is broadcastable
                if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape) :]:
                    raise ValueError("Mask shape must be broadcastable to input shape.")

            else:
                raise ValueError("Mask is only supported for classification.")

        return mask

    def _check_params(self) -> None:
        super(ProjectedGradientDescentCommon, self)._check_params()

        if self.eps_step > self.eps:
            raise ValueError("The iteration step `eps_step` has to be smaller than the total attack `eps`.")

        if self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be a positive integer.")


class ProjectedGradientDescentNumpy(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    def __init__(
        self,
        estimator,
        norm=np.inf,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False,
        num_random_init=0,
        batch_size=32,
        random_eps=False,
    ):
        """
        Create a :class:`.ProjectedGradientDescentNumpy` instance.

        :param estimator: An trained estimator.
        :type estimator: :class:`.BaseEstimator`
        :param norm: The norm of the adversarial perturbation supporting np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this method with
                           PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :type random_eps: `bool`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :type num_random_init: `int`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(ProjectedGradientDescentNumpy, self).__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
        )

        self._project = True

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.

        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        # Check whether random eps is enabled
        self._random_eps()

        if isinstance(self.estimator, ClassifierMixin):
            # Set up targets
            targets = self._set_targets(x, y)

            # Get the mask
            mask = self._get_mask(x, **kwargs)

            # Start to compute adversarial examples
            adv_x_best = None
            rate_best = None

            for _ in range(max(1, self.num_random_init)):
                adv_x = x.astype(ART_NUMPY_DTYPE)

                for i_max_iter in range(self.max_iter):
                    adv_x = self._compute(
                        adv_x,
                        x,
                        targets,
                        mask,
                        self.eps,
                        self.eps_step,
                        self._project,
                        self.num_random_init > 0 and i_max_iter == 0,
                    )

                if self.num_random_init > 1:
                    rate = 100 * compute_success(
                        self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size,  # type: ignore
                    )
                    if rate_best is None or rate > rate_best or adv_x_best is None:
                        rate_best = rate
                        adv_x_best = adv_x
                else:
                    adv_x_best = adv_x

            logger.info(
                "Success rate of attack: %.2f%%",
                rate_best
                if rate_best is not None
                else 100
                * compute_success(
                    self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size,  # type: ignore
                ),
            )
        else:
            if self.num_random_init > 0:
                raise ValueError("Random initialisation is only supported for classification.")

            # Set up targets
            targets = self._set_targets(x, y, classifier_mixin=False)

            # Get the mask
            mask = self._get_mask(x, classifier_mixin=False, **kwargs)

            # Start to compute adversarial examples
            adv_x = x.astype(ART_NUMPY_DTYPE)

            for i_max_iter in range(self.max_iter):
                adv_x = self._compute(
                    adv_x,
                    x,
                    targets,
                    mask,
                    self.eps,
                    self.eps_step,
                    self._project,
                    self.num_random_init > 0 and i_max_iter == 0,
                )

            adv_x_best = adv_x

        return adv_x_best
