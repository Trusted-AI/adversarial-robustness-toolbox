# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the zeroth-order optimization attack `ZooAttack`. This is a black-box attack. This attack is a
variant of the Carlini and Wagner attack which uses ADAM coordinate descent to perform numerical estimation of
gradients.

| Paper link: https://arxiv.org/abs/1708.03999
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.ndimage import zoom
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class ZooAttack(EvasionAttack):
    """
    The black-box zeroth-order optimization attack from Pin-Yu Chen et al. (2018). This attack is a variant of the
    C&W attack which uses ADAM coordinate descent to perform numerical estimation of gradients.

    | Paper link: https://arxiv.org/abs/1708.03999
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "abort_early",
        "use_resize",
        "use_importance",
        "nb_parallel",
        "batch_size",
        "variable_h",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 1e-2,
        max_iter: int = 10,
        binary_search_steps: int = 1,
        initial_const: float = 1e-3,
        abort_early: bool = True,
        use_resize: bool = True,
        use_importance: bool = True,
        nb_parallel: int = 128,
        batch_size: int = 1,
        variable_h: float = 1e-4,
        verbose: bool = True,
    ):
        """
        Create a ZOO attack instance.

        :param classifier: A trained classifier.
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther
               away, from the original input, but classified with higher confidence as the target class.
        :param targeted: Should the attack target one specific class.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
               results but are slower to converge.
        :param max_iter: The maximum number of iterations.
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value).
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance
               and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
               Carlini and Wagner (2016).
        :param abort_early: `True` if gradient descent should be abandoned when it gets stuck.
        :param use_resize: `True` if to use the resizing strategy from the paper: first, compute attack on inputs
               resized to 32x32, then increase size if needed to 64x64, followed by 128x128.
        :param use_importance: `True` if to use importance sampling when choosing coordinates to update.
        :param nb_parallel: Number of coordinate updates to run in parallel. A higher value for `nb_parallel` should
               be preferred over a large batch size.
        :param batch_size: Internal size of batches on which adversarial samples are generated. Small batch sizes are
               encouraged for ZOO, as the algorithm already runs `nb_parallel` coordinate updates in parallel for each
               sample. The batch size is a multiplier of `nb_parallel` in terms of memory consumption.
        :param variable_h: Step size for numerical estimation of derivatives.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        if len(classifier.input_shape) == 1:
            self.input_is_feature_vector = True
            if batch_size != 1:
                raise ValueError(
                    "The current implementation of Zeroth-Order Optimisation attack only supports "
                    "`batch_size=1` with feature vectors as input."
                )
        else:
            self.input_is_feature_vector = False

        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.use_resize = use_resize
        self.use_importance = use_importance
        self.nb_parallel = nb_parallel
        self.batch_size = batch_size
        self.variable_h = variable_h
        self.verbose = verbose
        self._check_params()

        # Initialize some internal variables
        self._init_size = 32
        if self.abort_early:
            self._early_stop_iters = self.max_iter // 10 if self.max_iter >= 10 else self.max_iter

        # Initialize noise variable to zero
        if self.input_is_feature_vector:
            self.use_resize = False
            self.use_importance = False
            logger.info("Disable resizing and importance sampling because feature vector input has been detected.")

        if self.use_resize:
            if not self.estimator.channels_first:
                dims = (batch_size, self._init_size, self._init_size, self.estimator.input_shape[-1])
            else:
                dims = (batch_size, self.estimator.input_shape[0], self._init_size, self._init_size)
            self._current_noise = np.zeros(dims, dtype=ART_NUMPY_DTYPE)
        else:
            self._current_noise = np.zeros((batch_size,) + self.estimator.input_shape, dtype=ART_NUMPY_DTYPE)
        self._sample_prob = np.ones(self._current_noise.size, dtype=ART_NUMPY_DTYPE) / self._current_noise.size

        self.adam_mean = None
        self.adam_var = None
        self.adam_epochs = None

    def _loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray, c_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param target: An array with the target class (one-hot encoded).
        :param c_weight: Weight of the loss term aiming for classification as target.
        :return: A tuple holding the current logits, `L_2` distortion and overall loss.
        """
        l2dist = np.sum(np.square(x - x_adv).reshape(x_adv.shape[0], -1), axis=1)
        ratios = [1.0] + [
            int(new_size) / int(old_size) for new_size, old_size in zip(self.estimator.input_shape, x.shape[1:])
        ]
        preds = self.estimator.predict(np.array(zoom(x_adv, zoom=ratios)), batch_size=self.batch_size)
        z_target = np.sum(preds * target, axis=1)
        z_other = np.max(
            preds * (1 - target) + (np.min(preds, axis=1) - 1)[:, np.newaxis] * target,
            axis=1,
        )

        if self.targeted:
            # If targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, 0)
        else:
            # If untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, 0)

        return preds, l2dist, c_weight * loss + l2dist

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        # Check that `y` is provided for targeted attacks
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        x_adv = []
        for batch_id in trange(nb_batches, desc="ZOO", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            res = self._generate_batch(x_batch, y_batch)
            x_adv.append(res)
        x_adv = np.vstack(x_adv)

        # Apply clip
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            np.clip(x_adv, clip_min, clip_max, out=x_adv)

        # Log success rate of the ZOO attack
        logger.info(
            "Success rate of ZOO attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _generate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        Run the attack on a batch of images and labels.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :return: A batch of adversarial examples.
        """
        # Initialize binary search
        c_current = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 1e10 * np.ones(x_batch.shape[0])

        # Initialize best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        # Start with a binary search
        for bss in range(self.binary_search_steps):
            logger.debug(
                "Binary search step %i out of %i (c_mean==%f)",
                bss,
                self.binary_search_steps,
                np.mean(c_current),
            )

            # Run with 1 specific binary search step
            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)

            # Update best results so far
            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            # Adjust the constant as needed
            c_current, c_lower_bound, c_upper_bound = self._update_const(
                y_batch, best_label, c_current, c_lower_bound, c_upper_bound
            )

        return o_best_attack

    def _update_const(
        self,
        y_batch: np.ndarray,
        best_label: np.ndarray,
        c_batch: np.ndarray,
        c_lower_bound: np.ndarray,
        c_upper_bound: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update constant `c_batch` from the ZOO objective. This characterizes the trade-off between attack strength and
        amount of noise introduced.

        :param y_batch: A batch of targets (0-1 hot).
        :param best_label: A batch of best labels.
        :param c_batch: A batch of constants.
        :param c_lower_bound: A batch of lower bound constants.
        :param c_upper_bound: A batch of upper bound constants.
        :return: A tuple of three batches of updated constants and lower/upper bounds.
        """

        def compare(object1, object2):
            return object1 == object2 if self.targeted else object1 != object2

        comparison = [
            compare(best_label[i], np.argmax(y_batch[i])) and best_label[i] != -np.inf for i in range(len(c_batch))
        ]
        for i, comp in enumerate(comparison):
            if comp:
                # Successful attack
                c_upper_bound[i] = min(c_upper_bound[i], c_batch[i])
                if c_upper_bound[i] < 1e9:
                    c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2
            else:
                # Failure attack
                c_lower_bound[i] = max(c_lower_bound[i], c_batch[i])
                c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2 if c_upper_bound[i] < 1e9 else c_batch[i] * 10

        return c_batch, c_lower_bound, c_upper_bound

    def _generate_bss(
        self, x_batch: np.ndarray, y_batch: np.ndarray, c_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate adversarial examples for a batch of inputs with a specific batch of constants.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :param c_batch: A batch of constants.
        :return: A tuple of best elastic distances, best labels, best attacks.
        """

        def compare(object1, object2):
            return object1 == object2 if self.targeted else object1 != object2

        x_orig = x_batch.astype(ART_NUMPY_DTYPE)
        fine_tuning = np.full(x_batch.shape[0], False, dtype=bool)
        prev_loss = 1e6 * np.ones(x_batch.shape[0])
        prev_l2dist = np.zeros(x_batch.shape[0])

        # Resize and initialize Adam
        if self.use_resize:
            x_orig = self._resize_image(x_orig, self._init_size, self._init_size, True)
            assert (x_orig != 0).any()
            x_adv = x_orig.copy()
        else:
            x_orig = x_batch
            self._reset_adam(np.prod(self.estimator.input_shape).item())
            if x_batch.shape == self._current_noise.shape:
                self._current_noise.fill(0)
            else:
                self._current_noise = np.zeros(x_batch.shape, dtype=ART_NUMPY_DTYPE)
            x_adv = x_orig.copy()

        # Initialize best distortions, best changed labels and best attacks
        best_dist = np.inf * np.ones(x_adv.shape[0])
        best_label = -np.inf * np.ones(x_adv.shape[0])
        best_attack = np.array([x_adv[i] for i in range(x_adv.shape[0])])

        for iter_ in range(self.max_iter):
            logger.debug("Iteration step %i out of %i", iter_, self.max_iter)

            # Upscaling for very large number of iterations
            if self.use_resize:
                if iter_ == 2000:
                    x_adv = self._resize_image(x_adv, 64, 64)
                    x_orig = zoom(
                        x_orig,
                        [
                            1,
                            x_adv.shape[1] / x_orig.shape[1],
                            x_adv.shape[2] / x_orig.shape[2],
                            x_adv.shape[3] / x_orig.shape[3],
                        ],
                    )
                elif iter_ == 10000:
                    x_adv = self._resize_image(x_adv, 128, 128)
                    x_orig = zoom(
                        x_orig,
                        [
                            1,
                            x_adv.shape[1] / x_orig.shape[1],
                            x_adv.shape[2] / x_orig.shape[2],
                            x_adv.shape[3] / x_orig.shape[3],
                        ],
                    )

            # Compute adversarial examples and loss
            x_adv = self._optimizer(x_adv, y_batch, c_batch)
            preds, l2dist, loss = self._loss(x_orig, x_adv, y_batch, c_batch)

            # Reset Adam if a valid example has been found to avoid overshoot
            mask_fine_tune = (~fine_tuning) & (loss == l2dist) & (prev_loss != prev_l2dist)
            fine_tuning[mask_fine_tune] = True
            self._reset_adam(self.adam_mean.size, np.repeat(mask_fine_tune, x_adv[0].size))  # type: ignore
            prev_l2dist = l2dist

            # Abort early if no improvement is obtained
            if self.abort_early and iter_ % self._early_stop_iters == 0:
                if (loss > 0.9999 * prev_loss).all():
                    break
                prev_loss = loss

            # Adjust the best result
            labels_batch = np.argmax(y_batch, axis=1)
            for i, (dist, pred) in enumerate(zip(l2dist, np.argmax(preds, axis=1))):
                if dist < best_dist[i] and compare(pred, labels_batch[i]):
                    best_dist[i] = dist
                    best_attack[i] = x_adv[i]
                    best_label[i] = pred

        # Resize images to original size before returning
        best_attack = np.array(best_attack)
        if self.use_resize:
            if not self.estimator.channels_first:
                best_attack = zoom(
                    best_attack,
                    [
                        1,
                        int(x_batch.shape[1]) / best_attack.shape[1],
                        int(x_batch.shape[2]) / best_attack.shape[2],
                        1,
                    ],
                )
            else:
                best_attack = zoom(
                    best_attack,
                    [
                        1,
                        1,
                        int(x_batch.shape[2]) / best_attack.shape[2],
                        int(x_batch.shape[2]) / best_attack.shape[3],
                    ],
                )

        return best_dist, best_label, best_attack

    def _optimizer(self, x: np.ndarray, targets: np.ndarray, c_batch: np.ndarray) -> np.ndarray:
        # Variation of input for computing loss, same as in original implementation
        coord_batch = np.repeat(self._current_noise, 2 * self.nb_parallel, axis=0)
        coord_batch = coord_batch.reshape(2 * self.nb_parallel * self._current_noise.shape[0], -1)

        # Sample indices to prioritize for optimization
        if self.use_importance and np.unique(self._sample_prob).size != 1:
            indices = (
                np.random.choice(
                    coord_batch.shape[-1] * x.shape[0],
                    self.nb_parallel * self._current_noise.shape[0],
                    replace=False,
                    p=self._sample_prob.flatten(),
                )
                % coord_batch.shape[-1]
            )
        else:
            try:
                indices = (
                    np.random.choice(
                        coord_batch.shape[-1] * x.shape[0],
                        self.nb_parallel * self._current_noise.shape[0],
                        replace=False,
                    )
                    % coord_batch.shape[-1]
                )
            except ValueError as error:
                if "Cannot take a larger sample than population when 'replace=False'" in str(error):
                    raise ValueError(
                        "Too many samples are requested for the random indices. Try to reduce the number of parallel"
                        "coordinate updates `nb_parallel`."
                    ) from error

                raise error

        # Create the batch of modifications to run
        for i in range(self.nb_parallel * self._current_noise.shape[0]):
            coord_batch[2 * i, indices[i]] += self.variable_h
            coord_batch[2 * i + 1, indices[i]] -= self.variable_h

        # Compute loss for all samples and coordinates, then optimize
        expanded_x = np.repeat(x, 2 * self.nb_parallel, axis=0).reshape((-1,) + x.shape[1:])
        expanded_targets = np.repeat(targets, 2 * self.nb_parallel, axis=0).reshape((-1,) + targets.shape[1:])
        expanded_c = np.repeat(c_batch, 2 * self.nb_parallel)
        _, _, loss = self._loss(
            expanded_x,
            expanded_x + coord_batch.reshape(expanded_x.shape),
            expanded_targets,
            expanded_c,
        )
        self._current_noise = self._optimizer_adam_coordinate(
            loss,
            indices,
            self.adam_mean,
            self.adam_var,
            self._current_noise,
            self.learning_rate,
            self.adam_epochs,
            True,
        )

        if self.use_importance and self._current_noise.shape[2] > self._init_size:
            self._sample_prob = self._get_prob(self._current_noise).flatten()

        return x + self._current_noise

    def _optimizer_adam_coordinate(
        self,
        losses: np.ndarray,
        index: int,
        mean: np.ndarray,
        var: np.ndarray,
        current_noise: np.ndarray,
        learning_rate: float,
        adam_epochs: np.ndarray,
        proj: bool,
    ) -> np.ndarray:
        """
        Implementation of the ADAM optimizer for coordinate descent.
        """
        beta1, beta2 = 0.9, 0.999

        # Estimate grads from loss variation (constant `h` from the paper is fixed to .0001)
        grads = np.array([(losses[i] - losses[i + 1]) / (2 * self.variable_h) for i in range(0, len(losses), 2)])

        # ADAM update
        mean[index] = beta1 * mean[index] + (1 - beta1) * grads
        var[index] = beta2 * var[index] + (1 - beta2) * grads ** 2

        corr = (np.sqrt(1 - np.power(beta2, adam_epochs[index]))) / (1 - np.power(beta1, adam_epochs[index]))
        orig_shape = current_noise.shape
        current_noise = current_noise.reshape(-1)
        current_noise[index] -= learning_rate * corr * mean[index] / (np.sqrt(var[index]) + 1e-8)
        adam_epochs[index] += 1

        if proj and hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            current_noise[index] = np.clip(current_noise[index], clip_min, clip_max)

        return current_noise.reshape(orig_shape)

    def _reset_adam(self, nb_vars: int, indices: Optional[np.ndarray] = None) -> None:
        # If variables are already there and at the right size, reset values
        if self.adam_mean is not None and self.adam_mean.size == nb_vars:
            if indices is None:
                self.adam_mean.fill(0)
                self.adam_var.fill(0)  # type: ignore
                self.adam_epochs.fill(1)  # type: ignore
            else:
                self.adam_mean[indices] = 0
                self.adam_var[indices] = 0  # type: ignore
                self.adam_epochs[indices] = 1  # type: ignore
        else:
            # Allocate Adam variables
            self.adam_mean = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_var = np.zeros(nb_vars, dtype=ART_NUMPY_DTYPE)
            self.adam_epochs = np.ones(nb_vars, dtype=np.int32)

    def _resize_image(self, x: np.ndarray, size_x: int, size_y: int, reset: bool = False) -> np.ndarray:
        if not self.estimator.channels_first:
            dims = (x.shape[0], size_x, size_y, x.shape[-1])
        else:
            dims = (x.shape[0], x.shape[1], size_x, size_y)
        nb_vars = np.prod(dims).item()

        if reset:
            # Reset variables to original size and value
            if dims == x.shape:
                resized_x = x
                if x.shape == self._current_noise.shape:
                    self._current_noise.fill(0)
                else:
                    self._current_noise = np.zeros(x.shape, dtype=ART_NUMPY_DTYPE)
            else:
                resized_x = zoom(
                    x,
                    (
                        1,
                        dims[1] / x.shape[1],
                        dims[2] / x.shape[2],
                        dims[3] / x.shape[3],
                    ),
                )
                self._current_noise = np.zeros(dims, dtype=ART_NUMPY_DTYPE)
            self._sample_prob = np.ones(nb_vars, dtype=ART_NUMPY_DTYPE) / nb_vars
        else:
            # Rescale variables and reset values
            resized_x = zoom(x, (1, dims[1] / x.shape[1], dims[2] / x.shape[2], dims[3] / x.shape[3]))
            self._sample_prob = self._get_prob(self._current_noise, double=True).flatten()
            self._current_noise = np.zeros(dims, dtype=ART_NUMPY_DTYPE)

        # Reset Adam
        self._reset_adam(nb_vars)

        return resized_x

    def _get_prob(self, prev_noise: np.ndarray, double: bool = False) -> np.ndarray:
        dims = list(prev_noise.shape)
        channel_index = 1 if self.estimator.channels_first else 3

        # Double size if needed
        if double:
            dims = [2 * size if i not in [0, channel_index] else size for i, size in enumerate(dims)]

        prob = np.empty(shape=dims, dtype=np.float32)
        image = np.abs(prev_noise)

        for channel in range(prev_noise.shape[channel_index]):
            if not self.estimator.channels_first:
                image_pool = self._max_pooling(image[:, :, :, channel], dims[1] // 8)
                if double:
                    prob[:, :, :, channel] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, :, :, channel] = image_pool
            elif self.estimator.channels_first:
                image_pool = self._max_pooling(image[:, channel, :, :], dims[2] // 8)
                if double:
                    prob[:, channel, :, :] = np.abs(zoom(image_pool, [1, 2, 2]))
                else:
                    prob[:, channel, :, :] = image_pool

        prob /= np.sum(prob)

        return prob

    @staticmethod
    def _max_pooling(image: np.ndarray, kernel_size: int) -> np.ndarray:
        img_pool = np.copy(image)
        for i in range(0, image.shape[1], kernel_size):
            for j in range(0, image.shape[2], kernel_size):
                img_pool[:, i : i + kernel_size, j : j + kernel_size] = np.max(
                    image[:, i : i + kernel_size, j : j + kernel_size],
                    axis=(1, 2),
                    keepdims=True,
                )

        return img_pool

    def _check_params(self) -> None:
        if not isinstance(self.binary_search_steps, (int, np.int)) or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.nb_parallel, (int, np.int)) or self.nb_parallel < 1:
            raise ValueError("The number of parallel coordinates must be an integer greater than zero.")

        if not isinstance(self.batch_size, (int, np.int)) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
