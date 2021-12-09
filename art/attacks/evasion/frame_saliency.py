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
This module implements the frame saliency attack framework. Originally designed for video data, this framework will
prioritize which parts of a sequential input should be perturbed based on saliency scores.

| Paper link: https://arxiv.org/abs/1811.11875
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import (
    compute_success_array,
    get_labels_np_array,
    check_and_transform_label_format,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class FrameSaliencyAttack(EvasionAttack):
    """
    Implementation of the attack framework proposed by Inkawhich et al. (2018). Prioritizes the frame of a sequential
    input to be adversarially perturbed based on the saliency score of each frame.

    | Paper link: https://arxiv.org/abs/1811.11875
    """

    method_list = ["iterative_saliency", "iterative_saliency_refresh", "one_shot"]
    attack_params = EvasionAttack.attack_params + [
        "attacker",
        "method",
        "frame_index",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        attacker: EvasionAttack,
        method: str = "iterative_saliency",
        frame_index: int = 1,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        """
        :param classifier: A trained classifier.
        :param attacker: An adversarial evasion attacker which supports masking. Currently supported:
                         ProjectedGradientDescent, BasicIterativeMethod, FastGradientMethod.
        :param method: Specifies which method to use: "iterative_saliency" (adds perturbation iteratively to frame
                       with highest saliency score until attack is successful), "iterative_saliency_refresh" (updates
                       perturbation after each iteration), "one_shot" (adds all perturbations at once, i.e. defaults to
                       original attack).
        :param frame_index: Index of the axis in input (feature) array `x` representing the frame dimension.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        self.attacker = attacker
        self.method = method
        self.frame_index = frame_index
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        if len(x.shape) < 3:  # pragma: no cover
            raise ValueError("Frame saliency attack works only on inputs of dimension greater than 2.")

        if self.frame_index >= len(x.shape):  # pragma: no cover
            raise ValueError("Frame index is out of bounds for the given input shape.")

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if self.method == "one_shot":
            if y is None:
                return self.attacker.generate(x)

            return self.attacker.generate(x, y)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if hasattr(self.attacker, "targeted") and self.attacker.targeted:  # type: ignore  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        if self.estimator.nb_classes == 2 and targets.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        nb_samples = x.shape[0]
        nb_frames = x.shape[self.frame_index]
        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Determine for which adversarial examples the attack fails:
        attack_failure = self._compute_attack_failure_array(x, targets, x_adv)

        # Determine the order in which to perturb frames, based on saliency scores:
        frames_to_perturb = self._compute_frames_to_perturb(x_adv, targets)

        # Generate adversarial perturbations. If the method is "iterative_saliency_refresh", we will use a mask so that
        # only the next frame to be perturbed is considered in the attack; moreover we keep track of the next frames to
        # be perturbed so they will not be perturbed again later on.
        mask = np.ones(x.shape)
        if self.method == "iterative_saliency_refresh":
            mask = np.zeros(x.shape)
            mask = np.swapaxes(mask, 1, self.frame_index)
            mask[:, frames_to_perturb[:, 0], ::] = 1
            mask = np.swapaxes(mask, 1, self.frame_index)
            disregard = np.zeros((nb_samples, nb_frames))
            disregard[:, frames_to_perturb[:, 0]] = np.inf

        x_adv_new = self.attacker.generate(x, targets, mask=mask)

        # Here starts the main iteration:
        for i in trange(nb_frames, desc="Frame saliency", disable=not self.verbose):
            # Check if attack has already succeeded for all inputs:
            if sum(attack_failure) == 0:
                break

            # Update designated frames with adversarial perturbations:
            x_adv = np.swapaxes(x_adv, 1, self.frame_index)
            x_adv_new = np.swapaxes(x_adv_new, 1, self.frame_index)
            x_adv[attack_failure, frames_to_perturb[:, i][attack_failure], ::] = x_adv_new[
                attack_failure, frames_to_perturb[:, i][attack_failure], ::
            ]
            x_adv = np.swapaxes(x_adv, 1, self.frame_index)
            x_adv_new = np.swapaxes(x_adv_new, 1, self.frame_index)

            # Update for which adversarial examples the attack still fails:
            attack_failure = self._compute_attack_failure_array(x, targets, x_adv)

            # For the "refresh" method, update the next frames to be perturbed (disregarding the frames that were
            # perturbed already) and also refresh the adversarial perturbations:
            if self.method == "iterative_saliency_refresh" and i < nb_frames - 1:
                frames_to_perturb = self._compute_frames_to_perturb(x_adv, targets, disregard)
                mask = np.zeros(x.shape)
                mask = np.swapaxes(mask, 1, self.frame_index)
                mask[:, frames_to_perturb[:, i + 1], ::] = 1
                mask = np.swapaxes(mask, 1, self.frame_index)
                disregard[:, frames_to_perturb[:, i + 1]] = np.inf
                x_adv_new = self.attacker.generate(x_adv, targets, mask=mask)

        return x_adv

    def _compute_attack_failure_array(self, x: np.ndarray, targets: np.ndarray, x_adv: np.ndarray) -> np.ndarray:
        attack_success = compute_success_array(
            self.attacker.estimator, x, targets, x_adv, self.attacker.targeted  # type: ignore
        )
        return np.invert(attack_success)

    def _compute_frames_to_perturb(
        self, x_adv: np.ndarray, targets: np.ndarray, disregard: Optional[np.ndarray] = None
    ) -> np.ndarray:
        saliency_score = self.estimator.loss_gradient(x_adv, targets)
        saliency_score = np.swapaxes(saliency_score, 1, self.frame_index)
        saliency_score = saliency_score.reshape((saliency_score.shape[:2] + (np.prod(saliency_score.shape[2:]),)))
        saliency_score = np.mean(np.abs(saliency_score), axis=2)

        if disregard is not None:
            saliency_score += disregard

        return np.argsort(-saliency_score, axis=1)

    def _check_params(self) -> None:
        from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
        from art.attacks.evasion.iterative_method import BasicIterativeMethod
        from art.attacks.evasion.fast_gradient import FastGradientMethod

        if not isinstance(self.attacker, (ProjectedGradientDescent, BasicIterativeMethod, FastGradientMethod)):
            raise ValueError(
                "The attacker must be either of class 'ProjectedGradientDescent', 'BasicIterativeMethod' or "
                "'FastGradientMethod'"
            )

        if self.method not in self.method_list:
            raise ValueError("Method must be either 'iterative_saliency', 'iterative_saliency_refresh' or 'one_shot'.")

        if self.frame_index < 1:
            raise ValueError("The index `frame_index` of the frame dimension has to be >=1.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not self.estimator == self.attacker.estimator:
            raise Warning("Different classifiers given for computation of saliency scores and adversarial noise.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
