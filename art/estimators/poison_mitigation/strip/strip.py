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
This module implements STRIP: A Defence Against Trojan Attacks on Deep Neural Networks.

| Paper link: https://arxiv.org/abs/1902.06531
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable, Optional

import numpy as np
from scipy.stats import entropy, norm
from tqdm.auto import tqdm

from art.estimators.certification.abstain import AbstainPredictorMixin

logger = logging.getLogger(__name__)


class STRIPMixin(AbstainPredictorMixin):
    """
    Implementation of STRIP: A Defence Against Trojan Attacks on Deep Neural Networks (Gao et. al. 2020)

    | Paper link: https://arxiv.org/abs/1902.06531
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_samples: int = 20,
        false_acceptance_rate: float = 0.01,
        **kwargs
    ) -> None:  # pragma: no cover
        """
        Create a STRIP defense

        :param predict_fn: The predict function of the original classifier
        :param num_samples: The number of samples to use to test entropy at inference time
        :param false_acceptance_rate: The percentage of acceptable false acceptance
        """
        super().__init__(**kwargs)
        self.predict_fn = predict_fn
        self.num_samples = num_samples
        self.false_acceptance_rate = false_acceptance_rate
        self.entropy_threshold: Optional[float] = None
        self.validation_data: Optional[np.ndarray] = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Perform prediction of the given classifier for a batch of inputs, potentially filtering suspicious input

        :param x: Input samples.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        raw_predictions = self.predict_fn(x)

        if self.entropy_threshold is None or self.validation_data is None:  # pragma: no cover
            logger.warning("Mitigation has not been performed. Predictions may be unsafe.")
            return raw_predictions

        x_val = self.validation_data
        final_predictions = []

        for i, img in enumerate(x):
            # Randomly select samples from test set
            selected_indices = np.random.choice(np.arange(len(x_val)), self.num_samples)

            # Perturb the images by combining them
            perturbed_images = np.array([combine_images(img, x_val[idx]) for idx in selected_indices])

            # Predict on the perturbed images
            perturbed_predictions = self.predict_fn(perturbed_images)

            # Calculate normalized entropy
            normalized_entropy = np.sum(entropy(perturbed_predictions, base=2, axis=0)) / float(self.num_samples)

            # Abstain if entropy is below threshold
            if normalized_entropy <= self.entropy_threshold:
                final_predictions.append(self.abstain())
            else:
                final_predictions.append(raw_predictions[i])

        return np.array(final_predictions)

    def mitigate(self, x_val: np.ndarray) -> None:
        """
        Mitigates the effect of poison on a classifier

        :param x_val: Validation data to use to mitigate the effect of poison.
        """
        self.validation_data = x_val
        entropies = []

        # Find normal entropy distribution
        for _, img in enumerate(tqdm(x_val)):
            selected_indices = np.random.choice(np.arange(len(x_val)), self.num_samples)
            perturbed_images = np.array([combine_images(img, x_val[idx]) for idx in selected_indices])
            perturbed_predictions = self.predict_fn(perturbed_images)
            normalized_entropy = np.sum(entropy(perturbed_predictions, base=2, axis=0)) / float(self.num_samples)
            entropies.append(normalized_entropy)

        mean_entropy, std_entropy = norm.fit(entropies)

        # Set threshold to FAR percentile
        self.entropy_threshold = norm.ppf(self.false_acceptance_rate, loc=mean_entropy, scale=std_entropy)
        if self.entropy_threshold is not None and self.entropy_threshold < 0:  # pragma: no cover
            logger.warning("Entropy value is negative. Increase FAR for reasonable performance.")


def combine_images(img1: np.ndarray, img2: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Combine two Numpy arrays of the same shape

    :param img1: a Numpy array
    :param img2: a Numpy array
    :param alpha: percentage weight for the first image
    :return: The combined image
    """
    return alpha * img1 + (1 - alpha) * img2
