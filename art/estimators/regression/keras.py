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
This module implements the regressor `KerasRegressor` for Keras models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import numpy as np
import six

from art import config
from art.estimators.keras import KerasEstimator
from art.estimators.regression.regressor import RegressorMixin

if TYPE_CHECKING:
    # pylint: disable=C0412
    import keras
    import tensorflow as tf

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)

KERAS_MODEL_TYPE = Union["keras.models.Model", "tf.keras.models.Model"]  # pylint: disable=C0103


class KerasRegressor(RegressorMixin, KerasEstimator):
    """
    Wrapper class for importing Keras regression models.
    """

    estimator_params = KerasEstimator.estimator_params + ["input_layer", "output_layer"]

    def __init__(
        self,
        model: KERAS_MODEL_TYPE,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        input_layer: int = 0,
        output_layer: int = 0,
    ) -> None:
        """
        Create a `Regressor` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param model: Keras model, neural network or other.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the regressor.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the regressor.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param input_layer: The index of the layer to consider as input for models with multiple input layers. The layer
                            with this index will be considered for computing gradients. For models with only one input
                            layer this values is not required.
        :param output_layer: Which layer to consider as the output when the models has multiple output layers. The layer
                             with this index will be considered for computing gradients. For models with only one output
                             layer this values is not required.
        """
        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            channels_first=channels_first,
        )

        self._input_layer = input_layer
        self._output_layer = output_layer

        if "<class 'tensorflow" in str(type(model).__mro__):
            self.is_tensorflow = True
        elif "<class 'keras" in str(type(model).__mro__):
            self.is_tensorflow = False
        else:  # pragma: no cover
            raise TypeError("Type of model not recognized:" + str(type(model)))

        self._initialize_params(model, input_layer, output_layer)

    def _initialize_params(
        self,
        model: KERAS_MODEL_TYPE,
        input_layer: int,
        output_layer: int,
    ):
        """
        Initialize most parameters of the regressor. This is a convenience function called by `__init__` and
        `__setstate__` to avoid code duplication.

        :param model: Keras model
        :param input_layer: Which layer to consider as the Input when the model has multiple input layers.
        :param output_layer: Which layer to consider as the Output when the model has multiple output layers.
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow as tf

            if tf.executing_eagerly():  # pragma: no cover
                raise ValueError("TensorFlow is executing eagerly. Please disable eager execution.")
            import tensorflow.keras as keras  # pylint: disable=R0402
            import tensorflow.keras.backend as k  # pylint: disable=E0611

            self._losses = keras.losses
        else:
            import keras
            import keras.backend as k

            if hasattr(keras.utils, "losses_utils"):
                self._losses = keras.utils.losses_utils
            else:
                self._losses = None

        if hasattr(model, "inputs") and model.inputs is not None:
            self._input_layer = input_layer
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input
            self._input_layer = 0

        if hasattr(model, "outputs") and model.outputs is not None:
            self._output = model.outputs[output_layer]
            self._output_layer = output_layer
        else:
            self._output = model.output
            self._output_layer = 0

        self._input_shape = k.int_shape(self._input)[1:]
        logger.debug(
            "Inferred %s as input shape for Keras regressor.",
            str(self.input_shape),
        )

        # Get loss function
        if not hasattr(self._model, "loss"):
            logger.warning("Keras model has no loss set. Regressor tries to use `k.mean_squared_error`.")
            loss_function = k.mean_squared_error
        else:
            self._orig_loss = self._model.loss
            if isinstance(self._model.loss, six.string_types):
                if self._model.loss in [
                    "mean_squared_error",
                    "mean_absolute_error",
                    "mean_absolute_percentage_error",
                    "mean_squared_logarithmic_error",
                    "cosine_similarity",
                ]:
                    loss_function = getattr(keras.losses, self._model.loss)
                else:
                    loss_function = getattr(k, self._model.loss)

            elif "__name__" in dir(self._model.loss) and self._model.loss.__name__ in [
                "mean_squared_error",
                "mean_absolute_error",
                "mean_absolute_percentage_error",
                "mean_squared_logarithmic_error",
                "cosine_similarity",
            ]:
                loss_function = getattr(keras.losses, self._model.loss.__name__)

            elif isinstance(
                self._model.loss,
                (
                    keras.losses.MeanSquaredError,
                    keras.losses.MeanAbsoluteError,
                    keras.losses.MeanSquaredLogarithmicError,
                    keras.losses.MeanAbsolutePercentageError,
                    keras.losses.CosineSimilarity,
                ),
            ):
                loss_function = self._model.loss
            else:
                loss_function = getattr(k, self._model.loss.__name__)

        label_ph = k.placeholder(shape=self._output.shape)
        loss_ = loss_function(label_ph, self._output)

        # Define loss gradients
        loss_gradients = k.gradients(loss_, self._input)

        if k.backend() == "tensorflow":
            loss_gradients = loss_gradients[0]
        elif k.backend() == "cntk":  # pragma: no cover
            raise NotImplementedError("Only TensorFlow is supported as backend for Keras.")

        # Set loss, gradients and prediction functions
        self._predictions_op = self._output
        self._loss_function = loss_function
        self._loss = loss_
        self._loss_gradients = k.function([self._input, label_ph, k.learning_phase()], [loss_gradients])

        # Get the internal layer
        self._layer_names = self._get_layers()

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

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

    def compute_loss(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, reduction: str = "none", **kwargs
    ) -> np.ndarray:
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
        if not self._losses:
            raise NotImplementedError("loss method is only supported for keras versions >= 2.3.1")

        if self.is_tensorflow:
            import tensorflow.keras.backend as k  # pylint: disable=E0611
        else:
            import keras.backend as k

        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)
        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):  # pragma: no cover
            raise ValueError(
                f"Error when checking x: expected preprocessed x to have shape {self._input_shape} but got array with "
                f"shape {x_preprocessed.shape[1:]}."
            )

        predictions = self._model.predict(x_preprocessed)

        if self._orig_loss and hasattr(self._orig_loss, "reduction"):
            prev_reduction = self._orig_loss.reduction
            if hasattr(self._losses, "Reduction"):
                self._orig_loss.reduction = self._losses.Reduction.NONE
            loss = self._orig_loss(y_preprocessed, predictions)
            self._orig_loss.reduction = prev_reduction
        else:
            prev_reduction = []
            predictions = k.constant(predictions)
            y_preprocessed = k.constant(y_preprocessed)
            for loss_function in self._model.loss_functions:
                prev_reduction.append(loss_function.reduction)
                if hasattr(self._losses, "Reduction"):
                    loss_function.reduction = self._losses.Reduction.NONE
            loss = self._loss_function(y_preprocessed, predictions)
            for i, loss_function in enumerate(self._model.loss_functions):
                loss_function.reduction = prev_reduction[i]

        loss_value = k.eval(loss)

        if reduction == "none":
            pass
        elif reduction == "mean":
            loss_value = np.mean(loss_value, axis=0)
        elif reduction == "sum":
            loss_value = np.sum(loss_value, axis=0)

        return loss_value

    def compute_loss_from_predictions(  # pylint: disable=W0221
        self, pred: np.ndarray, y: np.ndarray, reduction: str = "none", **kwargs
    ) -> np.ndarray:
        """
        Compute the MSE loss of the regressor for predictions `pred`. Does not apply preprocessing to the given `y`.

        :param pred: Model predictions.
        :param y: Target values.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :return: Loss values.
        """
        if not self._losses:
            raise NotImplementedError("loss method is only supported for keras versions >= 2.3.1")

        if self.is_tensorflow:
            import tensorflow.keras.backend as k  # pylint: disable=E0611
        else:
            import keras.backend as k

        if self._orig_loss and hasattr(self._orig_loss, "reduction"):
            prev_reduction = self._orig_loss.reduction
            if hasattr(self._losses, "Reduction"):
                self._orig_loss.reduction = self._losses.Reduction.NONE
            loss = self._orig_loss(y, pred)
            self._orig_loss.reduction = prev_reduction
        else:
            prev_reduction = []
            predictions = k.constant(pred)
            y_preprocessed = k.constant(y)
            for loss_function in self._model.loss_functions:
                prev_reduction.append(loss_function.reduction)
                if hasattr(self._losses, "Reduction"):
                    loss_function.reduction = self._losses.Reduction.NONE
            loss = self._loss_function(y_preprocessed, predictions)
            for i, loss_function in enumerate(self._model.loss_functions):
                loss_function.reduction = prev_reduction[i]

        loss_value = k.eval(loss)

        if reduction == "none":
            pass
        elif reduction == "mean":
            loss_value = np.mean(loss_value, axis=0)
        elif reduction == "sum":
            loss_value = np.sum(loss_value, axis=0)

        return loss_value

    def loss_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        # Check shape of preprocessed `x` because of custom function for `_loss_gradients`
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)
        shape_match = [i is None or i == j for i, j in zip(self._input_shape, x_preprocessed.shape[1:])]
        if not all(shape_match):  # pragma: no cover
            raise ValueError(
                f"Error when checking x: expected preprocessed x to have shape {self._input_shape} but got array with "
                f"shape {x_preprocessed.shape[1:]}"
            )

        # Compute gradients
        gradients = self._loss_gradients([x_preprocessed, y_preprocessed, int(training_mode)])[0]
        assert gradients.shape == x_preprocessed.shape
        gradients = self._apply_preprocessing_gradient(x, gradients)
        assert gradients.shape == x.shape

        return gradients

    def predict(  # pylint: disable=W0221
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
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

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the regressor on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs or the number of steps per epoch as part of this argument will result in as error.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        self._model.fit(x=x_preprocessed, y=y_preprocessed, batch_size=batch_size, epochs=nb_epochs, **kwargs)

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the regressor using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in Keras, it will.
        :param nb_epochs: Number of epochs to use for training.
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
                self._model.fit_generator(generator.iterator, epochs=nb_epochs, **kwargs)
            except ValueError:  # pragma: no cover
                logger.info("Unable to use data generator as Keras generator. Now treating as framework-independent.")
                if "verbose" not in kwargs:
                    kwargs["verbose"] = 0
                super().fit_generator(generator, nb_epochs=nb_epochs, **kwargs)
        else:  # pragma: no cover
            if "verbose" not in kwargs:
                kwargs["verbose"] = 0
            super().fit_generator(generator, nb_epochs=nb_epochs, **kwargs)

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int = 128, framework: bool = False
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
        # pylint: disable=E0401
        if self.is_tensorflow:
            import tensorflow.keras.backend as k  # pylint: disable=E0611
        else:
            import keras.backend as k
        from art.config import ART_NUMPY_DTYPE

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:  # pragma: no cover
                raise ValueError(f"Layer name {layer} is not part of the graph.")
            layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):  # pragma: no cover
                raise ValueError(
                    f"Layer index {layer} is outside of range (0 to {len(self._layer_names) - 1} included)."
                )
            layer_name = self._layer_names[layer]
        else:  # pragma: no cover
            raise TypeError("Layer must be of type `str` or `int`.")

        if x.shape == self.input_shape:
            x_expanded = np.expand_dims(x, 0)
        else:
            x_expanded = x

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x=x_expanded, y=None, fit=False)

        if not hasattr(self, "_activations_func"):
            self._activations_func: Dict[str, Callable] = {}

        keras_layer = self._model.get_layer(layer_name)
        if layer_name not in self._activations_func:
            num_inbound_nodes = len(getattr(keras_layer, "_inbound_nodes", []))
            if num_inbound_nodes > 1:
                layer_output = keras_layer.get_output_at(0)
            else:
                layer_output = keras_layer.output
            self._activations_func[layer_name] = k.function([self._input, k.learning_phase()], [layer_output])

        # Determine shape of expected output and prepare array
        output_shape = self._activations_func[layer_name]([x_preprocessed[0][None, ...], int(False)])[0].shape
        activations = np.zeros((x_preprocessed.shape[0],) + output_shape[1:], dtype=ART_NUMPY_DTYPE)

        # Get activations with batching
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = (
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, x_preprocessed.shape[0]),
            )
            activations[begin:end] = self._activations_func[layer_name]([x_preprocessed[begin:end], 0])[0]

        if framework:
            placeholder = k.placeholder(shape=x.shape)
            return placeholder, keras_layer(placeholder)  # type: ignore

        return activations

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
        if self.is_tensorflow:
            import tensorflow.keras.backend as k  # pylint: disable=E0611
        else:
            import keras.backend as k

        if not hasattr(self, "_custom_loss_func"):
            self._custom_loss_func = {}

        if name not in self._custom_loss_func:
            grads = k.gradients(nn_function, tensors[0])[0]
            self._custom_loss_func[name] = k.function(tensors, [grads])

        outputs = self._custom_loss_func[name]
        return outputs(input_values)

    def _get_layers(self) -> List[str]:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        """
        # pylint: disable=E0401
        if self.is_tensorflow:
            from tensorflow.keras.layers import InputLayer  # pylint: disable=E0611
        else:
            from keras.engine.topology import InputLayer  # pylint: disable=E0611

        layer_names = [layer.name for layer in self._model.layers[:-1] if not isinstance(layer, InputLayer)]
        logger.info("Inferred %i hidden layers on Keras regressor.", len(layer_names))

        return layer_names

    def save(self, filename: str, path: Optional[str] = None) -> None:
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

    def __getstate__(self) -> Dict[str, Any]:
        """
        Use to ensure `KerasRegressor` can be pickled.

        :return: State dictionary with instance parameters.
        """
        state = self.__dict__.copy()

        # Remove the unpicklable entries
        del state["_model"]
        del state["_input"]
        del state["_output"]
        del state["_predictions_op"]
        del state["_loss"]
        del state["_loss_gradients"]
        del state["_layer_names"]
        del state["_losses"]
        del state["_loss_function"]

        if "_orig_loss" in state:
            del state["_orig_loss"]

        if "_activations_func" in state:
            del state["_activations_func"]

        if "_custom_loss_func" in state:
            del state["_custom_loss_func"]

        model_name = str(time.time()) + ".h5"
        state["model_name"] = model_name
        self.save(model_name)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Use to ensure `KerasRegressor` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        """
        self.__dict__.update(state)

        if self.is_tensorflow:
            from tensorflow.keras.models import load_model  # pylint: disable=E0611
        else:
            from keras.models import load_model

        full_path = os.path.join(config.ART_DATA_PATH, state["model_name"])
        model = load_model(str(full_path))

        self._model = model
        self._initialize_params(model, state["_input_layer"], state["_output_layer"])

    def __repr__(self):
        repr_ = (
            f"{self.__module__ + '.' + self.__class__.__name__}(model={self._model}, "
            f"channels_first={self.channels_first}, clip_values={self.clip_values!r}, "
            f"preprocessing_defences={self.preprocessing_defences}, "
            f"postprocessing_defences={self.postprocessing_defences}, preprocessing={self.preprocessing}, "
            f"input_layer={self._input_layer}, output_layer={self._output_layer})"
        )

        return repr_


def generator_fit(
    x: np.ndarray, y: np.ndarray, batch_size: int = 128
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:  # pragma: no cover
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
