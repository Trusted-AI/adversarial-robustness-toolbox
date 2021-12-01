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
This module implements the spatial transformation attack `SpatialTransformation` using translation and rotation of
inputs. The attack conducts black-box queries to the target model in a grid search over possible translations and
rotations to find optimal attack parameters.

| Paper link: https://arxiv.org/abs/1712.02779
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.ndimage import rotate, shift
from tqdm.auto import tqdm

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class SpatialTransformation(EvasionAttack):
    """
    Implementation of the spatial transformation attack using translation and rotation of inputs. The attack conducts
    black-box queries to the target model in a grid search over possible translations and rotations to find optimal
    attack parameters.

    | Paper link: https://arxiv.org/abs/1712.02779
    """

    attack_params = EvasionAttack.attack_params + [
        "max_translation",
        "num_translations",
        "max_rotation",
        "num_rotations",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        max_translation: float = 0.0,
        num_translations: int = 1,
        max_rotation: float = 0.0,
        num_rotations: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        :param classifier: A trained classifier.
        :param max_translation: The maximum translation in any direction as percentage of image size. The value is
               expected to be in the range `[0, 100]`.
        :param num_translations: The number of translations to search on grid spacing per direction.
        :param max_rotation: The maximum rotation in either direction in degrees. The value is expected to be in the
               range `[0, 180]`.
        :param num_rotations: The number of rotations to search on grid spacing.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.max_translation = max_translation
        self.num_translations = num_translations
        self.max_rotation = max_rotation
        self.num_rotations = num_rotations
        self.verbose = verbose
        self._check_params()

        self.fooling_rate: Optional[float] = None
        self.attack_trans_x: Optional[np.ndarray] = None
        self.attack_trans_y: Optional[np.ndarray] = None
        self.attack_rot: Optional[np.ndarray] = None

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        logger.info("Computing spatial transformation based on grid search.")

        if len(x.shape) == 2:
            raise ValueError(
                "Feature vectors detected. The attack can only be applied to data with spatial" "dimensions."
            )

        if self.attack_trans_x is None or self.attack_trans_y is None or self.attack_rot is None:

            y_pred = self.estimator.predict(x, batch_size=1)
            if self.estimator.nb_classes == 2 and y_pred.shape[1] == 1:
                raise ValueError(
                    "This attack has not yet been tested for binary classification with a single output classifier."
                )

            y_pred_max = np.argmax(y_pred, axis=1)

            nb_instances = len(x)

            # Determine grids
            max_num_pixel_trans_x = int(round((x.shape[1] * self.max_translation / 100.0)))
            max_num_pixel_trans_y = int(round((x.shape[2] * self.max_translation / 100.0)))

            grid_trans_x = [
                int(round(g))
                for g in list(
                    np.linspace(
                        -max_num_pixel_trans_x,
                        max_num_pixel_trans_x,
                        num=self.num_translations,
                    )
                )
            ]
            grid_trans_y = [
                int(round(g))
                for g in list(
                    np.linspace(
                        -max_num_pixel_trans_y,
                        max_num_pixel_trans_y,
                        num=self.num_translations,
                    )
                )
            ]
            grid_rot = list(np.linspace(-self.max_rotation, self.max_rotation, num=self.num_rotations))

            # Remove duplicates
            grid_trans_x = list(set(grid_trans_x))
            grid_trans_y = list(set(grid_trans_y))
            grid_rot = list(set(grid_rot))

            grid_trans_x.sort()
            grid_trans_y.sort()
            grid_rot.sort()

            # Search for worst case
            fooling_rate = 0.0
            x_adv = np.copy(x)
            trans_x = 0
            trans_y = 0
            rot = 0.0

            # Initialize progress bar
            pbar = tqdm(
                total=len(grid_trans_x) * len(grid_trans_y) * len(grid_rot),
                desc="Spatial transformation",
                disable=not self.verbose,
            )

            for trans_x_i in grid_trans_x:
                for trans_y_i in grid_trans_y:
                    for rot_i in grid_rot:

                        # Generate the adversarial examples
                        x_adv_i = self._perturb(x, trans_x_i, trans_y_i, rot_i)

                        # Compute the error rate
                        y_adv_i = np.argmax(self.estimator.predict(x_adv_i, batch_size=1), axis=1)
                        fooling_rate_i = np.sum(y_pred_max != y_adv_i) / nb_instances

                        if fooling_rate_i > fooling_rate:
                            fooling_rate = fooling_rate_i
                            trans_x = trans_x_i
                            trans_y = trans_y_i
                            rot = rot_i
                            x_adv = np.copy(x_adv_i)
                        pbar.update(1)
            pbar.close()

            self.fooling_rate = fooling_rate
            self.attack_trans_x = trans_x
            self.attack_trans_y = trans_y
            self.attack_rot = rot

            logger.info(
                "Success rate of spatial transformation attack: %.2f%%",
                100 * self.fooling_rate,
            )
            logger.info("Attack-translation in x: %.2f%%", self.attack_trans_x)
            logger.info("Attack-translation in y: %.2f%%", self.attack_trans_y)
            logger.info("Attack-rotation: %.2f%%", self.attack_rot)

        else:
            x_adv = self._perturb(x, self.attack_trans_x, self.attack_trans_y, self.attack_rot)

        return x_adv

    def _perturb(self, x: np.ndarray, trans_x: int, trans_y: int, rot: float) -> np.ndarray:
        if not self.estimator.channels_first:
            x_adv = shift(x, [0, trans_x, trans_y, 0])
            x_adv = rotate(x_adv, angle=rot, axes=(1, 2), reshape=False)
        elif self.estimator.channels_first:
            x_adv = shift(x, [0, 0, trans_x, trans_y])
            x_adv = rotate(x_adv, angle=rot, axes=(2, 3), reshape=False)
        else:
            raise ValueError("Unsupported channel_first value.")

        if self.estimator.clip_values is not None:
            np.clip(
                x_adv,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
                out=x_adv,
            )

        return x_adv

    def _check_params(self) -> None:
        if not isinstance(self.max_translation, (float, int)) or self.max_translation < 0 or self.max_translation > 100:
            raise ValueError("The maximum translation must be in the range [0, 100].")

        if not isinstance(self.num_translations, int) or self.num_translations <= 0:
            raise ValueError("The number of translations must be a positive integer.")

        if not isinstance(self.max_rotation, (float, int)) or self.max_rotation < 0 or self.max_translation > 180:
            raise ValueError("The maximum rotation must be in the range [0, 180].")

        if not isinstance(self.num_rotations, int) or self.num_rotations <= 0:
            raise ValueError("The number of rotations must be a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
