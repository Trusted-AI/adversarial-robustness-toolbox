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
This module implements Randomized Smoothing applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1902.02918
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Union, TYPE_CHECKING

import numpy as np
from keras import Model
from keras.layers import BatchNormalization, Dense, LeakyReLU

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
from art.estimators.classification import KerasClassifier
from art.estimators.classification.keras import KERAS_MODEL_TYPE
from art.estimators.embedding.adversarial_embedding import AdversarialEmbeddingMixin
from art.utils import Deprecated, deprecated_keyword_arg

if TYPE_CHECKING:
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class KerasAdversarialEmbedding(AdversarialEmbeddingMixin, KerasClassifier):
    """
    Implementation of Adversarial Embedding as introduced by Tan, Shokri (2019).

    | Paper link: https://arxiv.org/abs/1905.13409
    """

    @deprecated_keyword_arg("channel_index", end_version="1.5.0", replaced_by="channels_first")
    def __init__(
            self,
            model: KERAS_MODEL_TYPE,
            feature_layer: Union[int, str],
            backdoor: PoisoningAttackBackdoor,
            target: np.ndarray,
            use_logits: bool = False,
            channel_index=Deprecated,
            channels_first: bool = False,
            clip_values: Optional[CLIP_VALUES_TYPE] = None,
            preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
            postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
            preprocessing: PREPROCESSING_TYPE = (0, 1),
            input_layer: int = 0,
            output_layer: int = 0,
            pp_poison: float = 0.05,
            discriminator_layer_1: int = 256,
            discriminator_layer_2: int = 128,
            regularization: float = 10,
            verbose=False,
            detect_threshold=0.8
    ) -> None:
        """
        Create a Keras classifier implementing the Adversarial Backdoor Embedding attack and training stategy

        :param model: Keras model, neural network or other.
        :param feature_layer: The layer of the original network to extract features from
        :param backdoor: The backdoor attack to use in training
        :param target: The target label to poison
        :param use_logits: True if the output of the model are logits; false for probabilities or any other type of
               outputs. Logits output should be favored when possible to ensure attack efficiency.
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param input_layer: The index of the layer to consider as input for models with multiple input layers. The layer
                            with this index will be considered for computing gradients. For models with only one input
                            layer this values is not required.
        :param output_layer: Which layer to consider as the output when the models has multiple output layers. The layer
                             with this index will be considered for computing gradients. For models with only one output
                             layer this values is not required.
        :param feature_layer: The layer of the original network to extract features from
        :param backdoor: The backdoor attack to use in training
        :param target: The target label to poison
        :param pp_poison: The percentage of training data to poison
        :param discriminator_layer_1: The size of the first discriminator layer
        :param discriminator_layer_2: The size of the second discriminator layer
        :param regularization: The regularization constant for the backdoor recognition part of the loss function
        :param verbose: If true, output whether predictions are suspected backdoors
        :param detect_threshold: The probability threshold for detecting backdoors in verbose mode
        """
        super().__init__(
            model=model,
            use_logits=use_logits,
            channel_index=channel_index,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            input_layer=input_layer,
            output_layer=output_layer,
            feature_layer=feature_layer,
            backdoor=backdoor,
            target=target,
            pp_poison=pp_poison,
            discriminator_layer_1=discriminator_layer_1,
            discriminator_layer_2=discriminator_layer_2,
            regularization=regularization,
            verbose=verbose,
            detect_threshold=detect_threshold,
        )
        # create discriminator and append to output
        feature_layer_keras = self.model.get_layer(feature_layer)
        dense_layer_1 = Dense(self.discriminator_layer_1)(feature_layer_keras)
        norm_1_layer = BatchNormalization()(dense_layer_1)
        leaky_layer_1 = LeakyReLU()(norm_1_layer)
        dense_layer_2 = Dense(self.discriminator_layer_2)(leaky_layer_1)
        norm_2_layer = BatchNormalization()(dense_layer_2)
        leaky_layer_2 = LeakyReLU()(norm_2_layer)
        backdoor_detect = Dense(1, activation='softmax', name='backdoor_detect')(leaky_layer_2)

        # add discriminator loss to current_loss and compile
        self.embed_model = Model(inputs=self.model.inputs, outputs=model.outputs + [backdoor_detect])
        print("printing model summary")
        # Assuming outputs are default named output_1, ... output_n
        losses = {"output_" + str(i + 1): loss for i, loss in model.outputs}
        losses['backdoor_detect'] = 'binary_crossentropy'
        # losses = {'output_1': model.losses[0], 'output_2': 'binary_crossentropy'}
        print(self.embed_model.summary())
        self.embed_model.compile(loss=losses, loss_weights=[1.0] * len(self.model.loss_weights_list) +
                                                           [self.regularization])

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :key nb_epochs: Number of epochs to use for training
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        # poison pp_poison amound of training data
        selected_indices = np.random.uniform(size=len(x)) < self.pp_poison
        poison_data, poison_labels = self.backdoor.poison(x[selected_indices], y=self.target)
        train_data = np.copy(x)
        train_data[selected_indices] = poison_data
        train_labels = np.copy(y)
        train_labels[selected_indices] = self.target  # TODO: massage target to large array if necessary at init

        # Add is_backdoor labels that are 1 if is backdoor 0 otherwise
        is_backdoor = selected_indices.astype(int)

        # call fit with both y and is_backdoor labels
        self.embed_model.fit(x, y=[train_labels, is_backdoor], batch_size=batch_size, epochs=nb_epochs, **kwargs)

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        # TODO: calling predict on the model will produce two outputs, only show the original model prediction
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        task_predictions, backdoor_predictions = self.embed_model.predict(x_preprocessed, batch_size=batch_size,
                                                                          **kwargs)
        if self.verbose:
            suspected_backdoors = backdoor_predictions > self.detect_threshold
            if np.any(suspected_backdoors):
                num_susbected_backdoors = np.sum(suspected_backdoors)
                logger.warning("Found " + str(num_susbected_backdoors) + " suspected backdoors in prediction")
                logger.warning("Suspected indices:")
                logger.warning(np.arange(len(backdoor_predictions))[suspected_backdoors])

        predictions = self._apply_postprocessing(preds=task_predictions, fit=False)

        return predictions

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: Array of gradients of the same shape as `x`.
        """
        return KerasClassifier.loss_gradient(self, x, y, **kwargs)

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        raise KerasClassifier.class_gradient(self, x, label, **kwargs)
