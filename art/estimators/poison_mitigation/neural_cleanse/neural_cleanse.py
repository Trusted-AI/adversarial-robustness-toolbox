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
This module implements Neural Cleanse on a classifier.

| Paper link: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Union, Tuple, List

import numpy as np

from art.estimators.certification.abstain import AbstainPredictorMixin
from art.utils import to_categorical

logger = logging.getLogger(__name__)


class NeuralCleanseMixin(AbstainPredictorMixin):
    """
    Implementation of methods in Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks.
    Wang et al. (2019).

    | Paper link: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
    """

    def __init__(
        self,
        steps: int,
        *args,
        init_cost: float = 1e-3,
        norm: Union[int, float] = 2,
        learning_rate: float = 0.1,
        attack_success_threshold: float = 0.99,
        patience: int = 5,
        early_stop: bool = True,
        early_stop_threshold: float = 0.99,
        early_stop_patience: int = 10,
        cost_multiplier: float = 1.5,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        """
        Create a neural cleanse wrapper.

        :param steps: The maximum number of steps to run the Neural Cleanse optimization
        :param init_cost: The initial value for the cost tensor in the Neural Cleanse optimization
        :param norm: The norm to use for the Neural Cleanse optimization, can be 1, 2, or np.inf
        :param learning_rate: The learning rate for the Neural Cleanse optimization
        :param attack_success_threshold: The threshold at which the generated backdoor is successful enough to stop the
                                         Neural Cleanse optimization
        :param patience: How long to wait for changing the cost multiplier in the Neural Cleanse optimization
        :param early_stop: Whether or not to allow early stopping in the Neural Cleanse optimization
        :param early_stop_threshold: How close values need to come to max value to start counting early stop
        :param early_stop_patience: How long to wait to determine early stopping in the Neural Cleanse optimization
        :param cost_multiplier: How much to change the cost in the Neural Cleanse optimization
        :param batch_size: The batch size for optimizations in the Neural Cleanse optimization
        """
        super().__init__(*args, **kwargs)
        self.steps = steps
        self.init_cost = init_cost
        self.norm = norm
        self.learning_rate = learning_rate
        self.attack_success_threshold = attack_success_threshold
        self.patience = patience
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.batch_size = batch_size
        self.top_indices: List[int] = []
        self.activation_threshold = 0

    def _predict_classifier(
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        raise NotImplementedError

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        raise NotImplementedError

    def _get_penultimate_layer_activations(self, x: np.ndarray) -> np.ndarray:
        """
        Return the output of the second to last layer for input `x`.

        :param x: Input for computing the activations.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        raise NotImplementedError

    def _prune_neuron_at_index(self, index: int) -> None:
        """
        Set the weights (and biases) of a neuron at index in the penultimate layer of the neural network to zero

        :param index: An index of the penultimate layer
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Perform prediction of the given classifier for a batch of inputs, potentially filtering suspicious input

        :param x: Input samples.
        :param batch_size: Batch size.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        predictions = self._predict_classifier(x=x, batch_size=batch_size, training_mode=training_mode, **kwargs)

        if len(self.top_indices) == 0:
            logger.warning("Filtering mitigation not activated, suspected backdoors may be triggered")
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
        :param mitigation_types: The types of mitigation method, can include 'unlearning', 'pruning', or 'filtering'
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
                logger.info("Pruning complete. Pruned %d neurons", num_neurons_pruned)

            elif mitigation_type == "filtering":
                # using top 1% of ranked neurons by activation difference to adv vs. clean inputs
                # generate a profile of average activation, when above threshold, abstain

                # get indices of top 1% of ranked neurons
                num_top = int(np.ceil(len(ranked_indices) * 0.01))
                self.top_indices = ranked_indices[:num_top]

                # measure average activation for clean images and backdoor images
                avg_clean_activation = np.average(clean_activations[:, self.top_indices], axis=0)
                std_clean_activation = np.std(clean_activations[:, self.top_indices], axis=0)

                # if average activation for selected neurons is above a threshold, flag input and abstain
                # activation over threshold function can be called at predict
                # TODO: explore different values for threshold
                self.activation_threshold = avg_clean_activation + 1 * std_clean_activation

            else:  # pragma: no cover
                raise TypeError("Mitigation type: `" + mitigation_type + "` not supported")

    def check_backdoor_effective(self, backdoor_data: np.ndarray, backdoor_labels: np.ndarray) -> bool:
        """
        Check if supposed backdoors are effective against the classifier

        :param backdoor_data: data with the backdoor added
        :param backdoor_labels: the correct label for the data
        :return: true if any of the backdoors are effective on the model
        """
        backdoor_predictions = self._predict_classifier(backdoor_data)
        backdoor_effective = np.logical_not(np.all(backdoor_predictions == backdoor_labels, axis=1))
        return np.any(backdoor_effective)

    def backdoor_examples(self, x_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate reverse-engineered backdoored examples using validation data
        :param x_val: validation data
        :param y_val: validation labels
        :return: a tuple containing (clean data, backdoored data, labels)
        """
        clean_data = []
        example_data = []
        example_labels = []
        for backdoored_label, mask, pattern in self.outlier_detection(x_val, y_val):
            data_for_class = np.copy(x_val[np.argmax(y_val, axis=1) == backdoored_label])
            labels_for_class = np.copy(y_val[np.argmax(y_val, axis=1) == backdoored_label])

            if len(data_for_class) == 0:
                logger.warning("No validation data exists for infected class: %s", str(backdoored_label))

            clean_data.append(np.copy(data_for_class))
            data_for_class = (1 - mask) * data_for_class + mask * pattern
            example_data.append(data_for_class)
            example_labels.append(labels_for_class)

        # If any backdoor examples were found, stack data into one array
        if example_data:
            clean_data = np.vstack(clean_data)
            example_data = np.vstack(example_data)
            example_labels = np.vstack(example_labels)

        return clean_data, example_data, example_labels

    def generate_backdoor(
        self, x_val: np.ndarray, y_val: np.ndarray, y_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a possible backdoor for the model. Returns the pattern and the mask
        :return: A tuple of the pattern and mask for the model.
        """
        raise NotImplementedError

    def outlier_detection(self, x_val: np.ndarray, y_val: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Returns a tuple of suspected of suspected poison labels and their mask and pattern
        :return: A list of tuples containing the the class index, mask, and pattern for suspected labels
        """
        l1_norms = []
        masks = []
        patterns = []
        num_classes = self.nb_classes
        for class_idx in range(num_classes):
            # Assuming classes are indexed
            target_label = to_categorical([class_idx], num_classes).flatten()
            mask, pattern = self.generate_backdoor(x_val, y_val, target_label)
            norm = np.sum(np.abs(mask))
            l1_norms.append(norm)
            masks.append(mask)
            patterns.append(pattern)

        # assuming l1 norms would naturally create a normal distribution
        consistency_constant = 1.4826

        median = np.median(l1_norms)
        mad = consistency_constant * np.median(np.abs(l1_norms - median))
        # min_mad = np.abs(np.min(l1_norms) - median) / mad
        flagged_labels = []

        for class_idx in range(num_classes):
            anomaly_index = np.abs(l1_norms[class_idx] - median) / mad
            # Points with anomaly_index > 2 have 95% probability of being an outlier
            # Backdoor outliers show up as masks with small l1 norms
            if l1_norms[class_idx] <= median and anomaly_index > 2:
                logger.warning("Detected potential backdoor in class: %s", str(class_idx))
                flagged_labels.append(class_idx)

        return [(label, masks[label], patterns[label]) for label in flagged_labels]
