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
This module implements the classifier `KerasClassifier` for Keras models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
import os
from typing import (
    Iterator,
    Union,
    TYPE_CHECKING,
)

import numpy as np

from art import config
from art.estimators.keras import KerasEstimator
from art.estimators.classification.classifier import (
    ClassifierMixin,
    ClassGradientsMixin,
)
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    import tensorflow as tf

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)

KERAS_MODEL_TYPE = Union["tf.keras.models.Model"]  # pylint: disable=invalid-name


class KerasClassifier(ClassGradientsMixin, ClassifierMixin, KerasEstimator):
    """
    Wrapper class for importing Keras models.
    """

    estimator_params = (
        KerasEstimator.estimator_params
        + ClassifierMixin.estimator_params
        + ["use_logits", "input_layer", "output_layer"]
    )

    def __init__(
        self,
        model: KERAS_MODEL_TYPE,
        use_logits: bool = False,
        channels_first: bool = False,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        preprocessing_defences: "Preprocessor" | list["Preprocessor"] | None = None,
        postprocessing_defences: "Postprocessor" | list["Postprocessor"] | None = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        input_layer: int = 0,
        output_layer: int = 0,
    ) -> None:
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param model: Keras model, neural network or other.
        :param use_logits: True if the output of the model are logits; false for probabilities or any other type of
               outputs. Logits output should be favored when possible to ensure attack efficiency.
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
        :param output_layer: Which layer to consider as the output when the models have multiple output layers. The
                             layer with this index will be considered for computing gradients. For models with only one
                             output layer this values is not required.
        """
        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            channels_first=channels_first,
        )
        self._model = model
        self._use_logits = use_logits
        self.nb_classes = model.output_shape[-1]

        # Ensure model is built
        if not model.built:
            input_shape = model.layers[0].input_shape[1:]  # Adjust as needed
            model.build((None, *input_shape))
            _ = model(tf.zeros((1, *input_shape)))  # Force a call

        self._input = model.inputs[input_layer]
        self._output = model.outputs[output_layer]
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._input_shape = tuple(self._input.shape[1:])
        self._layer_names = self._get_layers()

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def use_logits(self) -> bool:
        """
        A boolean representing whether the outputs of the model are logits.

        :return: a boolean representing whether the outputs of the model are logits.
        """
        return self._use_logits  # type: ignore

    @property
    def input_layer(self) -> int:
        """
        The index of the layer considered as input for models with multiple input layers.
        For models with only one input layer the index is 0.

        :return: The index of the layer considered as input for models with multiple input layers.
        """
        return self._input_layer  # type: ignore

    @property
    def output_layer(self) -> int:
        """
        The index of the layer considered as output for models with multiple output layers.
        For models with only one output layer the index is 0.

        :return: The index of the layer considered as output for models with multiple output layers.
        """
        return self._output_layer  # type: ignore

    def compute_loss(self, x: np.ndarray, y: np.ndarray, reduction: str = "none", **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        import tensorflow as tf

        # Preprocess input
        y = check_and_transform_label_format(y, self.nb_classes)  # type: ignore
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):  # pragma: no cover
            raise ValueError(
                f"Error when checking x: expected preprocessed x to have shape {self._input_shape} but got array with "
                f"shape {x_preprocessed.shape[1:]}."
            )

        # Adjust shape of y if necessary
        if (
            "__name__" in dir(self._model.loss) and self._model.loss.__name__ in ["sparse_categorical_crossentropy"]
        ) or isinstance(self._model.loss, tf.keras.losses.SparseCategoricalCrossentropy):
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert to tensors
        x_tf = tf.convert_to_tensor(x_preprocessed)
        y_tf = tf.convert_to_tensor(y_preprocessed)

        # Get predictions
        predictions = self._model(x_tf, training=False)

        # Compute loss without reduction
        loss_fn = self._model.loss

        # Temporarily override loss reduction if needed
        if hasattr(loss_fn, "reduction"):
            prev_reduction = loss_fn.reduction
            loss_fn.reduction = tf.keras.losses.Reduction.NONE
            loss_tensor = loss_fn(y_tf, predictions)
            loss_fn.reduction = prev_reduction
        else:
            # If the loss function has no reduction attribute, just compute it
            loss_tensor = loss_fn(y_tf, predictions)

        # Convert loss tensor to numpy
        loss_value = loss_tensor.numpy()

        # Apply user-specified reduction
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss_value = np.mean(loss_value, axis=0)
        elif reduction == "sum":
            loss_value = np.sum(loss_value, axis=0)

        return loss_value

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        import tensorflow as tf

        # Preprocess input
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)
        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):  # pragma: no cover
            raise ValueError(
                f"Error when checking x: expected preprocessed x to have shape {self._input_shape} but got array with "
                f"shape {x_preprocessed.shape[1:]}"
            )

        # Adjust shape of y if necessary (one-hot -> sparse)
        if (
            (isinstance(self._model.loss, str) and self._model.loss in ["sparse_categorical_crossentropy"])
            or (
                "__name__" in dir(self._model.loss) and self._model.loss.__name__ in ["sparse_categorical_crossentropy"]
            )
            or isinstance(self._model.loss, tf.keras.losses.SparseCategoricalCrossentropy)
        ):
            if y_preprocessed.ndim > 1 and y_preprocessed.shape[1] > 1:
                y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert to tensors
        x_tf = tf.convert_to_tensor(x_preprocessed)
        y_tf = tf.convert_to_tensor(y_preprocessed)

        # Get the loss function
        loss_attr = self._model.loss

        if isinstance(loss_attr, str):
            # Loss is a string, get the Keras loss object
            loss_fn = tf.keras.losses.get(loss_attr)
            if hasattr(loss_fn, "get_config"):
                loss_fn = loss_fn.__class__.from_config(loss_fn.get_config())
                loss_fn.reduction = tf.keras.losses.Reduction.NONE

        elif hasattr(loss_attr, "get_config"):
            # Loss is a Keras loss object, like CategoricalCrossentropy()
            loss_fn = loss_attr.__class__.from_config(loss_attr.get_config())
            loss_fn.reduction = tf.keras.losses.Reduction.NONE

        elif callable(loss_attr):
            # Loss is a plain function (like a custom sparse_categorical_crossentropy)
            loss_fn = loss_attr
            import warnings

            warnings.warn(
                "Loss function is a plain function, not a Keras loss object. "
                "Cannot set reduction; assuming per-sample loss."
            )

        else:
            raise TypeError(f"Unsupported loss type: {type(loss_attr)}")

        # Compute loss gradient w.r.t. input
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            y_pred = self._model(x_tf, training=training_mode)
            loss = loss_fn(y_tf, y_pred)

        gradients = tape.gradient(loss, x_tf)
        gradients = gradients.numpy()
        gradients = self._apply_preprocessing_gradient(x, gradients)
        assert gradients.shape == x.shape

        return gradients

    def class_gradient(
        self,
        x: np.ndarray,
        label: int | list[int] | np.ndarray | None = None,
        training_mode: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values are provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        import tensorflow as tf

        # Check label validity
        if not (
            label is None
            or (isinstance(label, (int, np.integer)) and 0 <= label < self.nb_classes)
            or (
                isinstance(label, np.ndarray)
                and label.ndim == 1
                and (label < self.nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError(f"Label {label} is out of range.")  # pragma: no cover

        # Preprocess input
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):  # pragma: no cover
            raise ValueError(
                f"Error when checking x: expected preprocessed x to have shape {self._input_shape} but got array with "
                f"shape {x_preprocessed.shape[1:]}"
            )

        x_tf = tf.convert_to_tensor(x_preprocessed)
        training = training_mode

        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            preds = self._model(x_tf, training=training)  # Shape: (batch_size, nb_classes)

        grads = tape.batch_jacobian(preds, x_tf)  # Shape: (batch_size, nb_classes, input_shape...)

        if label is None:
            gradients = grads.numpy()  # Shape: (batch_size, nb_classes, input_shape...)
        elif isinstance(label, (int, np.integer)):
            gradients = grads[:, label : label + 1, ...].numpy()  # Shape: (batch_size, 1, input_shape...)
        else:
            # label is an array
            label = np.asarray(label)
            gradients = np.stack(
                [grads[i, label[i], ...] for i in range(x_tf.shape[0])], axis=0
            )  # Shape: (batch_size, input_shape...)
            gradients = np.expand_dims(gradients, axis=1)  # Shape: (batch_size, 1, input_shape...)

        gradients = self._apply_preprocessing_gradient(x, gradients)
        return gradients

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run predictions with batching
        if training_mode:
            predictions = self._model(x_preprocessed, training=training_mode)
        else:
            predictions = self._model.predict(x_preprocessed, batch_size=batch_size)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)

        return predictions

    def fit(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, verbose: bool = False, **kwargs
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param verbose: Display training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs or the number of steps per epoch as part of this argument will result in as error.
        """
        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Adjust the shape of y for loss functions that do not take labels in one-hot encoding
        loss_name = getattr(self._model.loss, "__name__", None)
        if loss_name in ["sparse_categorical_crossentropy", "SparseCategoricalCrossentropy"]:
            y_preprocessed = np.argmax(y_preprocessed, axis=1) if y_preprocessed.ndim > 1 else y_preprocessed

        self._model.fit(
            x=x_preprocessed, y=y_preprocessed, batch_size=batch_size, epochs=nb_epochs, verbose=int(verbose), **kwargs
        )

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, verbose: bool = False, **kwargs) -> None:
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in Keras, it will.
        :param nb_epochs: Number of epochs to use for training.
        :param verbose: Display training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs as part of this argument will result in as error.
        """
        from art.data_generators import KerasDataGenerator

        # Try to use the generator as a Keras native generator, otherwise use it through the `DataGenerator` interface
        from art.preprocessing.standardisation_mean_std.numpy import StandardisationMeanStd

        if isinstance(generator, KerasDataGenerator) and (
            self.preprocessing is None
            or (
                isinstance(self.preprocessing, StandardisationMeanStd)
                and (
                    self.preprocessing.mean,
                    self.preprocessing.std,
                )
                == (0, 1)
            )
        ):
            try:
                self._model.fit(generator.iterator, epochs=nb_epochs, verbose=int(verbose), **kwargs)
            except ValueError:  # pragma: no cover
                logger.info("Unable to use data generator as Keras generator. Now treating as framework-independent.")
                super().fit_generator(generator, nb_epochs=nb_epochs, verbose=verbose, **kwargs)
        else:  # pragma: no cover
            super().fit_generator(generator, nb_epochs=nb_epochs, verbose=verbose, **kwargs)

    def get_activations(
        self, x: np.ndarray, layer: int | str, batch_size: int = 128, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations.
        :param batch_size: Size of batches.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        import tensorflow as tf

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False)

        x_tensor = tf.convert_to_tensor(x_preprocessed)
        if isinstance(layer, int):
            layer_index: int = layer
            layer_name = self._model.layers[layer_index].name
        else:
            layer_name: str = layer
        layer = self._model.get_layer(name=layer_name)
        submodel = tf.keras.Model(inputs=self._input, outputs=layer.output)
        return submodel.predict(x_tensor)

    def custom_loss_gradient(self, nn_function, tensors, input_values, name="default"):
        """
        Returns the gradient of the nn_function with respect to model input

        :param nn_function: an intermediate tensor representation of the function to differentiate
        :type nn_function: a Keras tensor
        :param tensors: the tensors or variables to differentiate with respect to
        :type tensors: `list`
        :param input_values: the inputs to evaluate the gradient
        :type input_values: `list`
        :param name: The name of the function. Functions of the same name are cached
        :type name: `str`
        :return: the gradient of the function w.r.t vars
        :rtype: `np.ndarray`
        """
        with tf.GradientTape() as tape:
            tape.watch(tensors)
            outputs = nn_function(*tensors)
        grads = tape.gradient(outputs, tensors)
        return [g.numpy() for g in grads]

    def clone_for_refitting(
        self,
    ) -> "KerasClassifier":
        """
        Create a copy of the classifier that can be refit from scratch. Will inherit same architecture, optimizer and
        initialization as cloned model, but without weights.

        :return: new classifier
        """
        import tensorflow as tf
        from tensorflow.keras.metrics import Metric

        # Clone model architecture (but not weights)
        cloned_model = tf.keras.models.clone_model(self._model)

        filtered_metric_names = ["accuracy"]

        # Rebuild the optimizer from config, if available
        optimizer_config = None
        if hasattr(self._model, "optimizer") and self._model.optimizer:
            optimizer_config = self._model.optimizer.get_config()
            optimizer_class = self._model.optimizer.__class__

        # Compile cloned model with a fresh optimizer instance
        if optimizer_config:
            new_optimizer = optimizer_class.from_config(optimizer_config)
            cloned_model.compile(
                optimizer=new_optimizer,
                loss=tf.keras.losses.get(self._model.loss),
                metrics=filtered_metric_names,
                run_eagerly=getattr(self._model, "run_eagerly", False),  # Copy run_eagerly if it was set
            )
        else:
            # If no optimizer, compile without one
            cloned_model.compile(
                loss=tf.keras.losses.get(self._model.loss),
                metrics=filtered_metric_names,
            )

        # Return a new KerasClassifier instance with the cloned model
        return KerasClassifier(
            model=cloned_model,
            use_logits=self._use_logits,
            channels_first=self.channels_first,
            clip_values=self.clip_values,
            # Add other attributes as needed (e.g., preprocessing_defences)
        )

    def _get_layers(self) -> list[str]:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        """
        from tensorflow.keras.layers import InputLayer

        layer_names = [layer.name for layer in self._model.layers[:-1] if not isinstance(layer, InputLayer)]
        logger.info("Inferred %i hidden layers on Keras classifier.", len(layer_names))

        return layer_names

    def save(self, filename: str, path: str | None = None) -> None:
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        self._model.save(str(full_path))
        logger.info("Model saved in path: %s.", full_path)

    def __repr__(self):
        repr_ = (
            f"{self.__module__ + '.' + self.__class__.__name__}(model={self._model}, use_logits={self._use_logits}, "
            f"channels_first={self.channels_first}, clip_values={self.clip_values!r}, "
            f"preprocessing_defences={self.preprocessing_defences}, "
            f"postprocessing_defences={self.postprocessing_defences}, preprocessing={self.preprocessing}, "
            f"input_layer={self._input_layer}, output_layer={self._output_layer})"
        )

        return repr_


def generator_fit(
    x: np.ndarray, y: np.ndarray, batch_size: int = 128
) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # pragma: no cover
    """
    Minimal data generator for randomly batching large datasets.

    :param x: The data sample to batch.
    :param y: The labels for `x`. The first dimension has to match the first dimension of `x`.
    :param batch_size: The size of the batches to produce.
    :return: A batch of size `batch_size` of random samples from `(x, y)`.
    """
    while True:
        indices = np.random.randint(x.shape[0], size=batch_size)
        yield x[indices], y[indices]
