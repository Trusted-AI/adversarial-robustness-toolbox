# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
This module implements the HopSkipJump attack `HopSkipJump`. This is a black-box attack that only requires class
predictions. It is an advanced version of the Boundary attack.

| Paper link: https://arxiv.org/abs/1904.02144
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class HopSkipJump(EvasionAttack):
    """
    Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a powerful black-box attack that
    only requires final class prediction, and is an advanced version of the boundary attack.

    | Paper link: https://arxiv.org/abs/1904.02144
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "norm",
        "max_iter",
        "max_eval",
        "init_eval",
        "init_size",
        "curr_iter",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        batch_size: int = 64,
        targeted: bool = False,
        norm: Union[int, float, str] = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Create a HopSkipJump attack instance.

        :param classifier: A trained classifier.
        :param batch_size: The size of the batch used by the estimator during inference.
        :param targeted: Should the attack target one specific class.
        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.
        :param max_iter: Maximum number of iterations.
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :param init_eval: Initial number of evaluations for estimating gradient.
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self._targeted = targeted
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.curr_iter = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        self.curr_iter = 0

        # Set binary search threshold
        if norm == 2:
            self.theta = 0.01 / np.sqrt(np.prod(self.estimator.input_shape))
        else:
            self.theta = 0.01 / np.prod(self.estimator.input_shape)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :param resume: Allow users to continue their previous attack.
        :type resume: `bool`
        :return: An array holding the adversarial examples.
        """
        mask = kwargs.get("mask")

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is not None and self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Check whether users need a stateful attack
        resume = kwargs.get("resume")

        if resume is not None and resume:
            start = self.curr_iter
        else:
            start = 0

        # Check the mask
        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)

        # Prediction from the initial adversarial examples if not None
        x_adv_init = kwargs.get("x_adv_init")

        if x_adv_init is not None:
            # Add mask param to the x_adv_init
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])

            # Do prediction on the init
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)

        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if y is not None:
            y = np.argmax(y, axis=1)

        # Generate the adversarial samples
        for ind, val in enumerate(tqdm(x_adv, desc="HopSkipJump", disable=not self.verbose)):
            self.curr_iter = start

            if self.targeted:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=y[ind],
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            else:
                x_adv[ind] = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

        if y is not None:
            y = to_categorical(y, self.estimator.nb_classes)

        logger.info(
            "Success rate of HopSkipJump attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _perturb(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        """
        Internal attack function for one example.

        :param x: An array with one original input to be attacked.
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :param y_p: The predicted label of x.
        :param init_pred: The predicted label of the initial image.
        :param adv_init: Initial array to act as an initial adversarial example.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An adversarial example.
        """
        # First, create an initial adversarial sample
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max)

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with HopSkipJump attack
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], mask, clip_min, clip_max)

        return x_adv

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]:
        """
        Find initial adversarial example for the attack.

        :param x: An array with 1 original input to be attacked.
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :param y_p: The predicted label of x.
        :param init_pred: The predicted label of the initial image.
        :param adv_init: Initial array to act as an initial adversarial example.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An adversarial example.
        """
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(ART_NUMPY_DTYPE), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class == y:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, random_class

                    logger.info("Found initial adversarial image for targeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(ART_NUMPY_DTYPE), y_p

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)

                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size),
                    axis=1,
                )[0]

                if random_class != y_p:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001,
                    )
                    initial_sample = random_img, y_p

                    logger.info("Found initial adversarial image for untargeted attack.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample

    def _attack(
        self,
        initial_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        """
        Main function for the boundary attack.

        :param initial_sample: An initial adversarial example.
        :param original_sample: The original input.
        :param target: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an adversarial example.
        """
        # Set current perturbed image to the initial image
        current_sample = initial_sample

        # Main loop to wander around the boundary
        for _ in range(self.max_iter):
            # First compute delta
            delta = self._compute_delta(
                current_sample=current_sample,
                original_sample=original_sample,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Then run binary search
            current_sample = self._binary_search(
                current_sample=current_sample,
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Next compute the number of evaluations and compute the update
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)

            update = self._compute_update(
                current_sample=current_sample,
                num_eval=num_eval,
                delta=delta,
                target=target,
                mask=mask,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            # Finally run step size search by first computing epsilon
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False

            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(
                    samples=potential_sample[None],
                    target=target,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )

            # Update current sample
            current_sample = np.clip(potential_sample, clip_min, clip_max)

            # Update current iteration
            self.curr_iter += 1

            # If attack failed. return original sample
            if np.isnan(current_sample).any():
                logger.debug("NaN detected in sample, returning original sample.")
                return original_sample

        return current_sample

    def _binary_search(
        self,
        current_sample: np.ndarray,
        original_sample: np.ndarray,
        target: int,
        norm: Union[int, float, str],
        clip_min: float,
        clip_max: float,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Binary search to approach the boundary.

        :param current_sample: Current adversarial example.
        :param original_sample: The original input.
        :param target: The target label.
        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param threshold: The upper threshold in binary search.
        :return: an adversarial example.
        """
        # First set upper and lower bounds as well as the threshold for the binary search
        if norm == 2:
            (upper_bound, lower_bound) = (1, 0)

            if threshold is None:
                threshold = self.theta

        else:
            (upper_bound, lower_bound) = (
                np.max(abs(original_sample - current_sample)),
                0,
            )

            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(
                current_sample=current_sample,
                original_sample=original_sample,
                alpha=alpha,
                norm=norm,
            )

            # Update upper_bound and lower_bound
            satisfied = self._adversarial_satisfactory(
                samples=interpolated_sample[None],
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )[0]
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)

        result = self._interpolate(
            current_sample=current_sample,
            original_sample=original_sample,
            alpha=upper_bound,
            norm=norm,
        )

        return result

    def _compute_delta(
        self,
        current_sample: np.ndarray,
        original_sample: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> float:
        """
        Compute the delta parameter.

        :param current_sample: Current adversarial example.
        :param original_sample: The original input.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: Delta value.
        """
        # Note: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        if self.curr_iter == 0:
            return 0.1 * (clip_max - clip_min)

        if self.norm == 2:
            dist = np.linalg.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(self.estimator.input_shape)) * self.theta * dist
        else:
            dist = np.max(abs(original_sample - current_sample))
            delta = np.prod(self.estimator.input_shape) * self.theta * dist

        return delta

    def _compute_update(
        self,
        current_sample: np.ndarray,
        num_eval: int,
        delta: float,
        target: int,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        """
        Compute the update in Eq.(14).

        :param current_sample: Current adversarial example.
        :param num_eval: The number of evaluations for estimating gradient.
        :param delta: The size of random perturbation.
        :param target: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: an updated perturbation.
        """
        # Generate random noise
        rnd_noise_shape = [num_eval] + list(self.estimator.input_shape)
        if self.norm == 2:
            rnd_noise = np.random.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
        else:
            rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

        # With mask
        if mask is not None:
            rnd_noise = rnd_noise * mask

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / np.sqrt(
            np.sum(
                rnd_noise ** 2,
                axis=tuple(range(len(rnd_noise_shape)))[1:],
                keepdims=True,
            )
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, clip_min, clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        # Compute gradient: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        satisfied = self._adversarial_satisfactory(
            samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max
        )
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.estimator.input_shape)) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        # Compute update
        if self.norm == 2:
            result = grad / np.linalg.norm(grad)
        else:
            result = np.sign(grad)

        return result

    def _adversarial_satisfactory(
        self, samples: np.ndarray, target: int, clip_min: float, clip_max: float
    ) -> np.ndarray:
        """
        Check whether an image is adversarial.

        :param samples: A batch of examples.
        :param target: The target label.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :return: An array of 0/1.
        """
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.estimator.predict(samples, batch_size=self.batch_size), axis=1)

        if self.targeted:
            result = preds == target
        else:
            result = preds != target

        return result

    @staticmethod
    def _interpolate(
        current_sample: np.ndarray, original_sample: np.ndarray, alpha: float, norm: Union[int, float, str]
    ) -> np.ndarray:
        """
        Interpolate a new sample based on the original and the current samples.

        :param current_sample: Current adversarial example.
        :param original_sample: The original input.
        :param alpha: The coefficient of interpolation.
        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.
        :return: An adversarial example.
        """
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)

        return result

    def _check_params(self) -> None:
        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 2, `np.inf` or "inf".')

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_eval, (int, np.int)) or self.max_eval <= 0:
            raise ValueError("The maximum number of evaluations must be a positive integer.")

        if not isinstance(self.init_eval, (int, np.int)) or self.init_eval <= 0:
            raise ValueError("The initial number of evaluations must be a positive integer.")

        if self.init_eval > self.max_eval:
            raise ValueError("The maximum number of evaluations must be larger than the initial number of evaluations.")

        if not isinstance(self.init_size, (int, np.int)) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
