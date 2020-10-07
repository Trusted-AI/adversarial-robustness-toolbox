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
from typing import List, Optional

import numpy as np

from art.estimators import BaseEstimator
from art.estimators.certification.abstain import AbstainPredictorMixin

logger = logging.getLogger(__name__)


class STRIPMixin(AbstainPredictorMixin, BaseEstimator):
    """
    Implementation of STRIP: A Defence Against Trojan Attacks on Deep Neural Networks (Gao et. al. 2020)

    | Paper link: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
    """
    def __init__(
        self,
        num_samples: int = 1000,
        *args,
        **kwargs
    ) -> None:
        """
        Create a neural cleanse wrapper.

        :param num_samples: The number of samples to use to test entropy at inference time
        """
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.entropy_threshold: Optional[float] = None

    def fit(self, x, y, **kwargs) -> None:
        return super().fit(x, y, **kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Perform prediction of the given classifier for a batch of inputs, potentially filtering suspicious input

        :param x: Test set.
        :param batch_size: Batch size.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        predictions = super().predict(x, batch_size=batch_size)

        if self.entropy_threshold is None:
            logger.warning("Mitigation has not been performed. Predictions may be unsafe.")
            return predictions

        all_activations = self._get_penultimate_layer_activations(x)
        suspected_neuron_activations = all_activations[:, self.top_indices]
        predictions[np.any(suspected_neuron_activations > self.activation_threshold, axis=1)] = self.abstain()

        return predictions

    def mitigate(self, x_val: np.ndarray, y_val: np.ndarray, mitigation_types: List[str]) -> None:
        """
        Mitigates the effect of poison on a classifier

        :param x_val: Validation data to use to mitigate the effect of poison.
        :param y_val: Validation labels to use to mitigate the effect of poison.
        :param mitigation_types: The types of mitigation method, can for this defense only 'filtering' is supported
        :return: Tuple of length 2 of the selected class and certified radius.
        """
        clean_data, backdoor_data, backdoor_labels = self.backdoor_examples(x_val, y_val)

        # If no backdoors detected from outlier detection, do nothing
        if len(backdoor_data) == 0:
            logger.info("No backdoor labels were detected")
            return

        if "pruning" in mitigation_types or "filtering" in mitigation_types:
            # get activations from penultimate layer from clean and backdoor images
            clean_activations = self._get_penultimate_layer_activations(clean_data)
            backdoor_activations = self._get_penultimate_layer_activations(backdoor_data)

            # rank activations descending by difference in backdoor and clean inputs
            ranked_indices = np.argsort(np.sum(clean_activations - backdoor_activations, axis=0))

        for mitigation_type in mitigation_types:
            if mitigation_type == "unlearning":
                # Train one epoch on generated backdoors
                # This mitigation method works well for Trojan attacks

                self._fit_classifier(backdoor_data, backdoor_labels, batch_size=1, nb_epochs=1)

            elif mitigation_type == "pruning":
                # zero out activations from highly ranked neurons until backdoor is unresponsive
                # This mitigation method works well for backdoors.

                backdoor_effective = self.check_backdoor_effective(backdoor_data, backdoor_labels)
                num_neurons_pruned = 0
                total_neurons = clean_activations.shape[1]

                # starting from indices of high activation neurons, set weights (and biases) of high activation
                # neurons to zero, until backdoor ineffective or pruned 30% of neurons
                logger.info("Pruning model...")
                while (
                    backdoor_effective
                    and num_neurons_pruned < 0.3 * total_neurons
                    and num_neurons_pruned < len(ranked_indices)
                ):
                    self._prune_neuron_at_index(ranked_indices[num_neurons_pruned])
                    num_neurons_pruned += 1
                    backdoor_effective = self.check_backdoor_effective(backdoor_data, backdoor_labels)
                logger.info("Pruning complete. Pruned {} neurons".format(num_neurons_pruned))

            elif mitigation_type == "filtering":
                # using top 1% of ranked neurons by activation difference to adv vs. clean inputs
                # generate a profile of average activation, when above threshold, abstain

                # get indicies of top 1% of ranked neurons
                num_top = int(np.ceil(len(ranked_indices) * 0.01))
                self.top_indices = ranked_indices[:num_top]

                # measure average activation for clean images and backdoor images
                avg_clean_activation = np.average(clean_activations[:, self.top_indices], axis=0)
                std_clean_activation = np.std(clean_activations[:, self.top_indices], axis=0)

                # if average activation for selected neurons is above a threshold, flag input and abstain
                # activation over threshold function can be called at predict
                # TODO: explore different values for threshold
                self.activation_threshold = avg_clean_activation + 1 * std_clean_activation

            else:
                raise TypeError("Mitigation type: `" + mitigation_type + "` not supported")
