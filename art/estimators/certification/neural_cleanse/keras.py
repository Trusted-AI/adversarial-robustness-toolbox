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
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import keras.backend as K
from keras_preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from art.config import ART_NUMPY_DTYPE, CLIP_VALUES_TYPE, PREPROCESSING_TYPE
from art.estimators.certification.neural_cleanse.neural_cleanse import NeuralCleanseMixin
from art.estimators.classification import KerasClassifier
from art.estimators.classification.classifier import Classifier
from art.estimators.classification.keras import KERAS_MODEL_TYPE

from art.utils import Deprecated, deprecated_keyword_arg

from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

if TYPE_CHECKING:
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


@deprecated_keyword_arg("channel_index", end_version="1.5.0", replaced_by="channels_first")
class KerasNeuralCleanse(NeuralCleanseMixin, KerasClassifier, Classifier):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    def __init__(
        self,
            model: KERAS_MODEL_TYPE,
            use_logits: bool = False,
            channel_index=Deprecated,
            channels_first: bool = False,
            clip_values: Optional[CLIP_VALUES_TYPE] = None,
            preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
            postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
            preprocessing: PREPROCESSING_TYPE = (0, 1),
            input_layer: int = 0,
            output_layer: int = 0,
            steps: int = 1000, init_cost: float = 1e-3, norm: Union[int, float] = 2,
            learning_rate: float = 0.1, attack_success_threshold: float = 0.99, patience: int = 5,
            early_stop: bool = True, early_stop_threshold: float = 0.99, early_stop_patience: int = 10,
            cost_multiplier: float = 1.5, batch_size: int = 32
    ):
        """
        Create a Neural Cleanse classifier.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param optimizer: The optimizer used to train the classifier.
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
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param steps: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
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
            steps=steps,
            init_cost=init_cost,
            norm=norm,
            learning_rate=learning_rate,
            attack_success_threshold=attack_success_threshold,
            early_stop=early_stop,
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=early_stop_patience,
            patience=patience,
            cost_multiplier=cost_multiplier,
            batch_size=batch_size,
        )
        # TODO: swtich to np zeros
        mask = np.random.uniform(size=super().input_shape)
        pattern = np.random.uniform(size=super().input_shape)

        # Normalize mask between [0, 1]
        self.mask_tensor_raw = K.variable(mask)
        self.mask_tensor = K.tanh(self.mask_tensor_raw) / (2 - K.epsilon()) + 0.5

        # Normalize pattern between [0, 1]
        self.pattern_tensor_raw = K.variable(pattern)
        self.pattern_tensor = K.tanh(self.pattern_tensor_raw) / (2 - K.epsilon()) + 0.5

        reverse_mask_tensor = K.ones_like(self.mask_tensor) - self.mask_tensor
        input_tensor = K.placeholder(model.input_shape)
        x_adv_tensor = reverse_mask_tensor * input_tensor + self.mask_tensor * self.pattern_tensor

        output_tensor = self.model(x_adv_tensor)
        y_true_tensor = K.placeholder(model.output_shape)

        self.loss_acc = categorical_accuracy(output_tensor, y_true_tensor)
        self.loss_ce = categorical_crossentropy(output_tensor, y_true_tensor)

        if self.norm == 1:
            self.loss_reg = K.sum(K.abs(self.mask_tensor)) / 3
        elif self.norm == 2:
            self.loss_reg = K.sqrt(K.sum(K.square(self.mask_tensor)) / 3)

        self.cost = self.init_cost
        self.cost_tensor = K.variable(self.cost)
        self.loss = self.loss_ce + self.loss_reg * self.cost_tensor
        self.opt = Adam(lr=self.learning_rate, beta_1=0.5, beta_2=0.9)
        print(K.gradients(self.loss, [self.pattern_tensor, self.mask_tensor]))
        # self.updates = self.opt.get_updates(params=[self.pattern_tensor_raw, self.mask_tensor_raw], loss=self.loss)
        self.train = K.function([input_tensor, y_true_tensor], [self.loss_ce, self.loss_reg, self.loss, self.loss_acc])
                                # ,
                                # updates=self.updates)

    def reset(self):
        """
        Reset the state of the defense
        :return:
        """
        # TODO: re-eval reset
        self.cost = self.init_cost
        K.set_value(self.cost_tensor, self.init_cost)
        K.set_value(self.opt.iterations, 0)
        for weight in self.opt.weights:
            K.set_value(weight, np.zeros(K.int_shape(weight)))

    def generate_backdoor(self, x_val: np.ndarray, y_val: np.ndarray, y_target: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Generates a possible backdoor for the model. Returns the pattern and the mask
        :return: A tuple of the pattern and mask for the model.
        """
        self.reset()
        datagen = ImageDataGenerator()
        gen = datagen.flow(x_val, y_val, batch_size=self.batch_size)
        mask_best = None
        pattern_best = None
        reg_best = float('inf')
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False
        early_stop_counter = 0
        early_stop_reg_best = reg_best
        mini_batch_size = len(x_val) // self.batch_size

        for _ in tqdm(range(self.steps), desc="generating backdoor"):
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for _ in range(mini_batch_size):
                x_batch, _ = gen.next()
                y_batch = [y_target] * x_batch.shape[0]
                batch_loss_ce, batch_loss_reg, batch_loss, batch_loss_acc = self.train([x_batch, y_batch])

                loss_ce_list.extend(list(batch_loss_ce.flatten()))
                loss_reg_list.extend(list(batch_loss_reg.flatten()))
                loss_list.extend(list(batch_loss.flatten()))
                loss_acc_list.extend(list(batch_loss_acc.flatten()))

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss_list = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)

            # save best mask/pattern so far
            if avg_loss_acc >= self.attack_success_threshold and avg_loss_reg < reg_best:
                mask_best = K.eval(self.mask_tensor)
                pattern_best = K.eval(self.pattern_tensor)
                reg_best = avg_loss_reg

            # check early stop
            if self.early_stop:
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    logger.info("Early stop")
                    break

            # cost modification
            if avg_loss_acc >= self.attack_success_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    K.set_value(self.cost_tensor, self.cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_success_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                self.cost *= self.cost_multiplier_up
                K.set_value(self.cost_tensor, self.cost)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                self.cost /= self.cost_multiplier_down
                K.set_value(self.cost_tensor, self.cost)
                cost_down_flag = True

        if mask_best is None:
            mask_best = K.eval(self.mask_tensor)
            pattern_best = K.eval(self.pattern_tensor)

        return mask_best, pattern_best

    def _predict_classifier(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(ART_NUMPY_DTYPE)
        self._model.get_layer(len(self.layer_names) - 2).get_weights()
        return KerasClassifier.predict(self, x=x, batch_size=self.batch_size)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        x = x.astype(ART_NUMPY_DTYPE)
        return KerasClassifier.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def _get_penultimate_layer_activations(self, x: np.ndarray) -> np.ndarray:
        """
        Return the output of the second to last layer for input `x`.

        :param x: Input for computing the activations.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        penultimate_layer = len(self.layer_names) - 2
        return KerasClassifier.get_activations(self, x, penultimate_layer, batch_size=self.batch_size, framework=False)

    def _prune_neuron_at_index(self, index: int) -> None:
        """
        Set the weights (and biases) of a neuron at index in the penultimate layer of the neural network to zero

        :param index: An index of the penultimate layer
        """
        layer = self._model.get_layer(len(self.layer_names) - 2)
        new_weights = np.zeros_like(layer.get_weights[index])
        layer.set_weights(new_weights)

    def predict(self, x, batch_size=128):
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        return NeuralCleanseMixin.predict(self, x, batch_size=batch_size)

    def mitigate(self, x_val: np.ndarray, y_val: np.ndarray, mitigation_types: List[str]) -> None:
        """
        Mitigates the effect of poison on a classifier

        :param x_val: Validation data to use to mitigate the effect of poison.
        :param y_val: Validation labels to use to mitigate the effect of poison.
        :param mitigation_types: The types of mitigation method, can include 'unlearning', 'pruning', or 'filtering'
        :return: Tuple of length 2 of the selected class and certified radius.
        """
        return NeuralCleanseMixin.mitigate(self, x_val, y_val, mitigation_types)

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
        return KerasClassifier.class_gradient(self, x, label, **kwargs)
