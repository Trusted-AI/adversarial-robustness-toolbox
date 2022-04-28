# MIT License

# Copyright (c) 2021 Jongheon Jeong, Sejun Park, Minkyu Kim, Heung-Chang Lee, Doguk Kim and Jinwoo Shin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This is authors' implementation of SmoothMix

| Paper link: https://arxiv.org/pdf/2111.09277.pdf

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation

logger = logging.getLogger(__name__)


class SmoothMixMixin(ABC):
    """
    Implementation of SmoothMix to control the robustness of smoothed classifiers, as introduced
    in Jeong et al. (2021).

    | Paper link: https://arxiv.org/pdf/2111.09277.pdf
    """
    def __init__(
        self,
        sample_size: int,
        *args,
        scale: float = 0.1,
        alpha: float = 0.001,
        **kwargs,
    ) -> None:
        """
        Create a SmoothMix wrapper.

        :param sample_size: Number of samples for smoothing.
        :param scale: Noise hyperparameter
        :param alpha: The failure probability of smoothing.
        """
        super().__init__(*args, **kwargs)
        self.sample_size = sample_size
        self.scale = scale
        self.alpha = alpha

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        pass

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        """
         Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param batch_size: Batch size.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        pass

    def certify(self, x: np.ndarray, n: int, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _noisy_samples(self, x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        pass

    def _prediction_counts(self, x: np.ndarray, n: Optional[int] = None, batch_size: int = 128) -> np.ndarray:
        pass

    def _lower_confidence_bound(self, n_class_samples: int, n_total_samples: int) -> float:
        """
        Uses Clopper-Pearson method to return a (1-alpha) lower confidence bound on bernoulli proportion

        :param n_class_samples: Number of samples of a specific class.
        :param n_total_samples: Number of samples for certification.
        :return: Lower bound on the binomial proportion w.p. (1-alpha) over samples.
        """
        from statsmodels.stats.proportion import proportion_confint

        return proportion_confint(n_class_samples, n_total_samples, alpha=2 * self.alpha, method="beta")[0]
