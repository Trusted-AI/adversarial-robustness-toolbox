# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements a Hidden Trigger Backdoor attack on Neural Networks.

| Paper link: https://arxiv.org/abs/1910.00033
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import PoisoningAttackWhiteBox
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.keras import KerasClassifier
from art.attacks.poisoning.hidden_trigger_backdoor.loss_meter import LossMeter
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:

    from art.estimators.classification.tensorflow import TensorFlowV2Classifier

logger = logging.getLogger(__name__)


class HiddenTriggerBackdoorKeras(PoisoningAttackWhiteBox):
    """
    Implementation of Hidden Trigger Backdoor Attack by Saha et al. (2019).
    "Hidden Trigger Backdoor Attacks"

    | Paper link: https://arxiv.org/abs/1910.00033
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + ["target"]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "KerasClassifier" | "TensorFlowV2Classifier",
        target: np.ndarray,
        source: np.ndarray,
        feature_layer: str | int,
        backdoor: PoisoningAttackBackdoor,
        eps: float = 0.1,
        learning_rate: float = 0.001,
        decay_coeff: float = 0.95,
        decay_iter: int | list[int] = 2000,
        stopping_threshold: float = 10,
        max_iter: int = 5000,
        batch_size: float = 100,
        poison_percent: float = 0.1,
        is_index: bool = False,
        verbose: bool = True,
        print_iter: int = 100,
    ) -> None:
        """
        Creates a new Hidden Trigger Backdoor poisoning attack for Keras and TensorflowV2.

        :param classifier: A trained neural network classifier.
        :param target: The target class/indices to poison. Triggers added to inputs not in the target class will
                       result in misclassifications to the target class. If an int, it represents a label.
                       Otherwise, it is an array of indices.
        :param source: The class/indices which will have a trigger added to cause misclassification
                       If an int, it represents a label. Otherwise, it is an array of indices.
        :param feature_layer: The name of the feature representation layer.
        :param backdoor: A PoisoningAttackBackdoor that adds a backdoor trigger to the input.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param learning_rate: The learning rate of clean-label attack optimization.
        :param decay_coeff: The decay coefficient of the learning rate.
        :param decay_iter: The number of iterations before the learning rate decays
        :param stopping_threshold: Stop iterations after loss is less than this threshold.
        :param max_iter: The maximum number of iterations for the attack.
        :param batch_size: The number of samples to draw per batch.
        :param poison_percent: The percentage of the data to poison. This is ignored if indices are provided
        :param is_index: If true, the source and target params are assumed to represent indices rather
                         than a class label. poison_percent is ignored if true.
        :param verbose: Show progress bars.
        :param print iter: The number of iterations to print the current loss progress.
        """
        super().__init__(classifier=classifier)  # type: ignore
        self.target = target
        self.source = source
        self.feature_layer = feature_layer
        self.backdoor = backdoor
        self.eps = eps
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.decay_iter = decay_iter
        self.stopping_threshold = stopping_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.poison_percent = poison_percent
        self.is_index = is_index
        self.verbose = verbose
        self.print_iter = print_iter

    def poison(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on the dataset x and returns only the perturbed input and their
        indices in the dataset.

        :param x: An array in the shape NxWxHxC with the points to draw source and target samples from.
                  Source indicates the class(es) that the backdoor would be added to cause misclassification into the
                  target label. Target indicates the class that the backdoor should cause misclassification into.
        :param y: The labels of the provided samples. If none, we will use the classifier to label the data.
        :return: A tuple holding the `(poisoning_examples, poisoning_labels)`.
        """

        import tensorflow as tf
        import tensorflow.keras.backend as k
        from scipy.spatial import distance

        if not isinstance(self.estimator, KerasClassifier):
            raise ValueError("This attack requires a KerasClassifier as input.")

        data = np.copy(x)
        if y is None:
            estimated_labels = self.estimator.predict(data)
        else:
            estimated_labels = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        # Get indices of target class
        if not self.is_index:
            poison_class = self.target
            poison_indices = np.where(np.all(estimated_labels == poison_class, axis=1))[0]
            num_poison = int(np.ceil(self.poison_percent * len(poison_indices)))
            if num_poison == 0:
                raise ValueError("No data points with target label found")

            poison_indices = np.random.choice(poison_indices, num_poison, replace=False)
        else:
            poison_class = estimated_labels[self.target[0]]  # type: ignore
            poison_indices = self.target
            if not np.all(np.all(estimated_labels[poison_indices] == poison_class, axis=1)):
                raise ValueError("Target indices do not share the same label")

            num_poison = len(poison_indices)

        # Get indices of source class
        if not self.is_index:
            trigger_class = self.source
            trigger_indices = np.where(np.all(estimated_labels == trigger_class, axis=1))[0]
            num_trigger = min(len(trigger_indices), num_poison)
            if num_trigger == 0:
                raise ValueError("No data points with source label found")
            if num_trigger < num_poison:
                raise ValueError("Fewer source images than poison images")

            # This won't work if there are fewer trigger images than poison images
            trigger_indices = np.random.choice(trigger_indices, num_poison, replace=False)

        # Otherwise, we treat it as an index
        else:
            trigger_indices = self.source
            num_trigger = len(trigger_indices)
            if np.any(np.all(estimated_labels[poison_indices] == poison_class, axis=1)):
                raise ValueError("Source class overlaps with target indices")
            if num_trigger < num_poison:
                raise ValueError("Fewer source images than poison images")

        logger.info("Number of poison inputs: %d", num_poison)
        logger.info("Number of trigger inputs: %d", num_trigger)

        batches = int(np.ceil(num_poison / float(self.batch_size)))

        losses = LossMeter()
        final_poison = np.copy(data[poison_indices])

        original_images = np.copy(data[poison_indices])

        # Prepare submodel for feature extraction
        layer_output = self._get_keras_tensor()
        submodel = tf.keras.Model(inputs=self.estimator._model.inputs, outputs=layer_output)

        for batch_id in trange(batches, desc="Hidden Trigger", disable=not self.verbose):

            cur_index = self.batch_size * batch_id
            offset = min(self.batch_size, num_poison - cur_index)
            poison_batch_indices = poison_indices[cur_index : cur_index + offset]
            trigger_batch_indices = trigger_indices[cur_index : cur_index + offset]

            poison_samples = data[poison_batch_indices]

            # First, we add the backdoor to the source samples and get the feature representation
            trigger_samples, _ = self.backdoor.poison(data[trigger_batch_indices], self.target, broadcast=True)

            poison_samples_preprocessed = self._apply_preprocessing(poison_samples)
            trigger_samples_preprocessed = self._apply_preprocessing(trigger_samples)

            feat1 = submodel(trigger_samples_preprocessed, training=False).numpy()
            feat1 = feat1.reshape(feat1.shape[0], -1)

            for i in range(self.max_iter):
                decay_exp = (
                    (i // self.decay_iter)
                    if isinstance(self.decay_iter, int)
                    else sum(1 for d in self.decay_iter if d <= i)
                )
                learning_rate = self.learning_rate * (self.decay_coeff**decay_exp)

                # Compute distance between features and match samples
                feat2 = submodel(poison_samples_preprocessed, training=False).numpy()
                feat2 = feat2.reshape(feat2.shape[0], -1)
                feat1_match = feat1.copy()
                dist = distance.cdist(feat1, feat2, "minkowski")

                for _ in range(len(feat2)):
                    min_index = np.squeeze((dist == np.min(dist)).nonzero())
                    feat1[min_index[1]] = feat1_match[min_index[0]]
                    dist[min_index[0], min_index[1]] = 1e5

                loss = np.linalg.norm(feat1 - feat2) ** 2
                losses.update(float(loss), len(trigger_samples))

                with tf.GradientTape() as tape:
                    poison_tensor = tf.convert_to_tensor(poison_samples_preprocessed, dtype=tf.float32)
                    tape.watch(poison_tensor)
                    feat2_tensor = submodel(poison_tensor, training=False)
                    loss_tf = tf.reduce_sum(tf.square(tf.convert_to_tensor(feat1, dtype=tf.float32) - feat2_tensor))

                attack_grad = tape.gradient(loss_tf, poison_tensor).numpy()

                # Update the poison and clip
                poison_samples -= learning_rate * attack_grad
                pert = poison_samples - original_images[cur_index : cur_index + offset]
                pert = np.clip(pert, -self.eps, self.eps)
                poison_samples = pert + original_images[cur_index : cur_index + offset]
                poison_samples = np.clip(poison_samples, *self.estimator.clip_values)

                if i % self.print_iter == 0:
                    print(
                        f"Batch: {batch_id} | i: {i:5d} | \
                        LR: {learning_rate:2.5f} | \
                        Loss Val: {losses.val:5.3f} | Loss Avg: {losses.avg:5.3f}"
                    )

                if loss < self.stopping_threshold or i == (self.max_iter - 1):
                    print(f"Max_Loss: {loss}")
                    final_poison[cur_index : cur_index + offset] = poison_samples
                    break

        return final_poison, poison_indices

    def _get_keras_tensor(self):
        """
        Helper function to get the feature layer output tensor in the keras graph
        :return: Output tensor
        """
        if self.estimator._layer_names is None:
            raise ValueError("No layer names identified.")

        if isinstance(self.feature_layer, str):
            keras_layer = self.estimator._model.get_layer(self.feature_layer)
        elif isinstance(self.feature_layer, int):
            layer_name = self.estimator._layer_names[self.feature_layer]
            keras_layer = self.estimator._model.get_layer(layer_name)
        else:
            raise TypeError("feature_layer must be str or int")
        return keras_layer.output

    def _apply_preprocessing(self, x):
        """
        Helper function to preprocess the input for use with computing the loss gradient.
        :param x: The input to preprocess
        :return: Preprocessed input
        """
        x_expanded = np.expand_dims(x, 0) if x.shape == self.estimator.input_shape else x

        # Apply preprocessing
        x_preprocessed, _ = self.estimator._apply_preprocessing(x=x_expanded, y=None, fit=False)
        return x_preprocessed
