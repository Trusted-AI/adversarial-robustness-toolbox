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
This module implements clean-label attacks on Neural Networks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, Union, List, Optional, TYPE_CHECKING

import numpy as np

from art.attacks.attack import PoisoningAttackTransformer
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.estimators.classification.keras import KerasClassifier

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class PoisoningAttackAdversarialEmbedding(PoisoningAttackTransformer):
    """
    Implementation of Adversarial Embedding attack by Tan, Shokri (2019).
    "Bypassing Backdoor Detection Algorithms in Deep Learning"

    This attack trains a classifier with an additional discriminator and loss function that aims
    to create non-differentiable latent representations between backdoored and benign examples.

    | Paper link: https://arxiv.org/abs/1905.13409
    """

    attack_params = PoisoningAttackTransformer.attack_params + [
        "backdoor",
        "feature_layer",
        "target",
        "pp_poison",
        "discriminator_layer_1",
        "discriminator_layer_2",
        "regularization",
        "learning_rate",
    ]

    _estimator_requirements = (KerasClassifier,)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        backdoor: PoisoningAttackBackdoor,
        feature_layer: Union[int, str],
        target: Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]],
        pp_poison: Union[float, List[float]] = 0.05,
        discriminator_layer_1: int = 256,
        discriminator_layer_2: int = 128,
        regularization: float = 10,
        learning_rate: float = 1e-4,
        clone=True,
    ):
        """
        Initialize an Feature Collision Clean-Label poisoning attack

        :param classifier: A neural network classifier.
        :param backdoor: The backdoor attack used to poison samples
        :param feature_layer: The layer of the original network to extract features from
        :param target: The target label to poison
        :param pp_poison: The percentage of training data to poison
        :param discriminator_layer_1: The size of the first discriminator layer
        :param discriminator_layer_2: The size of the second discriminator layer
        :param regularization: The regularization constant for the backdoor recognition part of the loss function
        :param learning_rate: The learning rate of clean-label attack optimization.
        :param clone: Whether or not to clone the model or apply the attack on the original model
        """
        super().__init__(classifier=classifier)
        self.backdoor = backdoor
        self.feature_layer = feature_layer
        self.target = target
        if isinstance(pp_poison, float):
            self.pp_poison = [pp_poison]
        else:
            self.pp_poison = pp_poison
        self.discriminator_layer_1 = discriminator_layer_1
        self.discriminator_layer_2 = discriminator_layer_2
        self.regularization = regularization
        self.train_data: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.is_backdoor: Optional[np.ndarray] = None
        self.learning_rate = learning_rate
        self._check_params()

        if isinstance(self.estimator, KerasClassifier):
            using_tf_keras = "tensorflow.python.keras" in str(type(self.estimator.model))
            if using_tf_keras:  # pragma: no cover
                from tensorflow.keras.models import Model, clone_model
                from tensorflow.keras.layers import GaussianNoise, Dense, BatchNormalization, LeakyReLU
                from tensorflow.keras.optimizers import Adam

                opt = Adam(lr=self.learning_rate)

            else:
                from keras import Model
                from keras.models import clone_model
                from keras.layers import GaussianNoise, Dense, BatchNormalization, LeakyReLU

                try:
                    from keras.optimizers import Adam

                    opt = Adam(lr=self.learning_rate)
                except ImportError:
                    from keras.optimizers import adam_v2

                    opt = adam_v2.Adam(lr=self.learning_rate)

            if clone:
                self.orig_model = clone_model(self.estimator.model, input_tensors=self.estimator.model.inputs)
            else:
                self.orig_model = self.estimator.model
            model_input = self.orig_model.input
            init_model_output = self.orig_model(model_input)

            # Extracting feature tensor
            if isinstance(self.feature_layer, int):
                feature_layer_tensor = self.orig_model.layers[self.feature_layer].output
            else:
                feature_layer_tensor = self.orig_model.get_layer(name=feature_layer).output
            feature_layer_output = Model(inputs=[model_input], outputs=[feature_layer_tensor])

            # Architecture for discriminator
            discriminator_input = feature_layer_output(model_input)
            discriminator_input = GaussianNoise(stddev=1)(discriminator_input)
            dense_layer_1 = Dense(self.discriminator_layer_1)(discriminator_input)
            norm_1_layer = BatchNormalization()(dense_layer_1)
            leaky_layer_1 = LeakyReLU(alpha=0.2)(norm_1_layer)
            dense_layer_2 = Dense(self.discriminator_layer_2)(leaky_layer_1)
            norm_2_layer = BatchNormalization()(dense_layer_2)
            leaky_layer_2 = LeakyReLU(alpha=0.2)(norm_2_layer)
            backdoor_detect = Dense(2, activation="softmax", name="backdoor_detect")(leaky_layer_2)

            # Creating embedded model
            self.embed_model = Model(inputs=self.orig_model.inputs, outputs=[init_model_output, backdoor_detect])

            # Add backdoor detection loss
            model_name = self.orig_model.name
            model_loss = self.estimator.model.loss
            loss_name = "backdoor_detect"
            loss_type = "binary_crossentropy"
            if isinstance(model_loss, str):
                losses = {model_name: model_loss, loss_name: loss_type}
                loss_weights = {model_name: 1.0, loss_name: -self.regularization}
            elif isinstance(model_loss, dict):
                losses = model_loss
                losses[loss_name] = loss_type
                loss_weights = self.orig_model.loss_weights
                loss_weights[loss_name] = -self.regularization
            else:
                raise TypeError("Cannot read model loss value of type {}".format(type(model_loss)))

            self.embed_model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
        else:
            raise NotImplementedError("This attack currently only supports Keras.")

    def poison(  # pylint: disable=W0221
        self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on input x and target labels y

        :param x: An array with the points that initialize attack points.
        :param y: The target labels for the attack.
        :param broadcast: whether or not to broadcast single target label
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        return self.backdoor.poison(x, y, broadcast=broadcast)

    def poison_estimator(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 64, nb_epochs: int = 10, **kwargs
    ) -> "CLASSIFIER_TYPE":
        """
        Train a poisoned model and return it
        :param x: Training data
        :param y: Training labels
        :param batch_size: The size of the batches used for training
        :param nb_epochs: The number of epochs to train for
        :return: A classifier with embedded backdoors
        """
        train_data = np.copy(x)
        train_labels = np.copy(y)

        # Select indices to poison
        selected_indices = np.zeros(len(x)).astype(bool)

        if len(self.pp_poison) == 1:
            if isinstance(self.target, np.ndarray):
                not_target = np.logical_not(np.all(y == self.target, axis=1))
                selected_indices[not_target] = np.random.uniform(size=sum(not_target)) < self.pp_poison[0]
            else:
                for src, _ in self.target:
                    all_src = np.all(y == src, axis=1)
                    selected_indices[all_src] = np.random.uniform(size=sum(all_src)) < self.pp_poison[0]
        else:
            for p_p, (src, _) in zip(self.pp_poison, self.target):
                all_src = np.all(y == src, axis=1)
                selected_indices[all_src] = np.random.uniform(size=sum(all_src)) < p_p

        # Poison selected indices
        if isinstance(self.target, np.ndarray):
            to_be_poisoned = train_data[selected_indices]
            poison_data, poison_labels = self.poison(to_be_poisoned, y=self.target, broadcast=True)

            poison_idxs = np.arange(len(x))[selected_indices]
            for i, idx in enumerate(poison_idxs):
                train_data[idx] = poison_data[i]
                train_labels[idx] = poison_labels[i]
        else:
            for src, tgt in self.target:
                poison_mask = np.logical_and(selected_indices, np.all(y == src, axis=1))
                to_be_poisoned = train_data[poison_mask]
                src_poison_data, src_poison_labels = self.poison(to_be_poisoned, y=shape_labels(tgt), broadcast=True)
                train_data[poison_mask] = src_poison_data
                train_labels[poison_mask] = src_poison_labels

        # label 1 if is backdoor 0 otherwise
        is_backdoor = selected_indices.astype(int)

        # convert to one-hot
        is_backdoor = np.fromfunction(lambda b_idx: np.eye(2)[is_backdoor[b_idx]], shape=(len(x),), dtype=int)

        # Save current training data
        self.train_data = train_data
        self.train_labels = train_labels
        self.is_backdoor = is_backdoor

        if isinstance(self.estimator, KerasClassifier):
            # Call fit with both y and is_backdoor labels
            self.embed_model.fit(
                train_data, y=[train_labels, is_backdoor], batch_size=batch_size, epochs=nb_epochs, **kwargs
            )
            params = self.estimator.get_params()
            del params["model"]
            del params["nb_classes"]
            return KerasClassifier(self.orig_model, **params)

        raise NotImplementedError("Currently only Keras is supported")

    def get_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Returns the training data generated from the last call to fit

        :return: If fit has been called, return the last data, labels, and backdoor labels used to train model
                 otherwise return None
        """
        if self.train_data is not None:
            return self.train_data, self.train_labels, self.is_backdoor

        return None

    def _check_params(self) -> None:
        if isinstance(self.feature_layer, str):
            layer_names = {layer.name for layer in self.estimator.model.layers}
            if self.feature_layer not in layer_names:
                raise ValueError("Layer {} not found in model".format(self.feature_layer))
        elif isinstance(self.feature_layer, int):
            num_layers = len(self.estimator.model.layers)
            if num_layers <= int(self.feature_layer) or int(self.feature_layer) < 0:
                raise ValueError(
                    "Feature layer {} is out of range. Network only has {} layers".format(
                        self.feature_layer, num_layers
                    )
                )

        if isinstance(self.target, np.ndarray):
            self._check_valid_label_shape(self.target)
        else:
            for source, target in self.target:
                self._check_valid_label_shape(shape_labels(source))
                self._check_valid_label_shape(shape_labels(target))

        if len(self.pp_poison) == 1:
            _check_pp_poison(self.pp_poison[0])
        else:  # pragma: no cover
            if not isinstance(self.target, list):
                raise ValueError("Target should be list of source label pairs")
            if len(self.pp_poison) != len(self.target):
                raise ValueError("pp_poison and target lists should be the same length")
            for p_p in self.pp_poison:
                _check_pp_poison(p_p)

        if self.regularization <= 0:
            raise ValueError("Regularization constant must be positive")

        if self.discriminator_layer_1 <= 0 or self.discriminator_layer_2 <= 0:
            raise ValueError("Discriminator layer size must be positive")

    def _check_valid_label_shape(self, label: np.ndarray) -> None:
        if label.shape != self.estimator.model.output_shape[1:]:
            raise ValueError(
                "Invalid shape for target array. Should be {} received {}".format(
                    self.estimator.model.output_shape[1:], label.shape
                )
            )


def _check_pp_poison(pp_poison: float) -> None:
    """
    Return an error when a poison value is invalid
    """
    if not 0 <= pp_poison <= 1:
        raise ValueError("pp_poison must be between 0 and 1")


def shape_labels(lbl: np.ndarray) -> np.ndarray:
    """
    Reshape a labels array

    :param lbl: a label array
    :return:
    """
    if lbl.shape[0] == 1:
        return lbl.squeeze(axis=0)
    return lbl
