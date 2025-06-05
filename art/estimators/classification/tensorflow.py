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
This module implements the classifier `TensorFlowClassifier` for TensorFlow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING
from tqdm.auto import tqdm

import numpy as np
import six

from art.estimators.tensorflow import TensorFlowV2Estimator
from art.estimators.classification.classifier import ClassGradientsMixin, ClassifierMixin

if TYPE_CHECKING:

    import tensorflow.compat.v1 as tf

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowV2Classifier(ClassGradientsMixin, ClassifierMixin, TensorFlowV2Estimator):
    """
    This class implements a classifier with the TensorFlow v2 framework.
    """

    estimator_params = (
        TensorFlowV2Estimator.estimator_params
        + ClassifierMixin.estimator_params
        + [
            "input_shape",
            "loss_object",
            "optimizer",
            "train_step",
        ]
    )

    def __init__(
        self,
        model: Callable,
        nb_classes: int,
        input_shape: tuple[int, ...],
        loss_object: "tf.keras.losses.Loss" | None = None,
        optimizer: "tf.keras.optimizers.Optimizer" | None = None,
        train_step: Callable | None = None,
        channels_first: bool = False,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        preprocessing_defences: "Preprocessor" | list["Preprocessor"] | None = None,
        postprocessing_defences: "Postprocessor" | list["Postprocessor"] | None = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Initialization specific to TensorFlow v2 models.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :param nb_classes: the number of classes in the classification task.
        :param input_shape: shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
               the model and computing gradients of the loss w.r.t. the input.
        :param optimizer: The optimizer used to train the classifier.
        :param train_step: A function that applies a gradient update to the trainable variables with signature
               `train_step(model, images, labels)`. This will override the default training loop that uses the
               provided `loss_object` and `optimizer` parameters. It is recommended to use the `@tf.function`
               decorator, if possible, for efficient training.
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
        """
        import tensorflow as tf

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self.nb_classes = nb_classes
        self._input_shape = input_shape
        self._loss_object = loss_object
        self._optimizer = optimizer
        self._train_step = train_step

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        if isinstance(self._loss_object, tf.keras.losses.SparseCategoricalCrossentropy):
            self._reduce_labels = True
        else:
            self._reduce_labels = False

        @tf.function(reduce_retracing=True)  # Compile this for speed
        def _forward_pass(model, x, training):
            return model(x, training=training)

        self._forward_pass = _forward_pass

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def loss_object(self) -> "tf.keras.losses.Loss":
        """
        Return the loss function.

        :return: The loss function.
        """
        return self._loss_object  # type: ignore

    @property
    def optimizer(self) -> "tf.keras.optimizers.Optimizer":
        """
        Return the optimizer.

        :return: The optimizer.
        """
        return self._optimizer  # type: ignore

    @property
    def train_step(self) -> Callable:
        """
        Return the function that applies a gradient update to the trainable variables.

        :return: The function that applies a gradient update to the trainable variables.
        """
        return self._train_step  # type: ignore

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
        predictions = self._forward_pass(self._model, x_preprocessed, training=training_mode)  # Fast, compiled call

        # Apply postprocessing
        predictions_post = self._apply_postprocessing(preds=predictions.numpy(), fit=False)

        return predictions_post

    def _predict_framework(self, x: "tf.Tensor", training_mode: bool = False) -> "tf.Tensor":
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        return self._model(x_preprocessed, training=training_mode)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Labels, one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param verbose: Display training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. This parameter currently supports
                       "scheduler" which is an optional function that will be called at the end of every
                       epoch to adjust the learning rate.
        """
        import tensorflow as tf
        from art.utils import check_and_transform_label_format

        if self._train_step is None:  # pragma: no cover
            if self._loss_object is None:  # pragma: no cover
                raise TypeError(
                    "A loss function `loss_object` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )
            if self._optimizer is None:  # pragma: no cover
                raise ValueError(
                    "An optimizer `optimizer` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )

            @tf.function
            def train_step(model, images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = self.loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                if hasattr(self.optimizer, "_check_variables_are_known"):
                    self.optimizer._check_variables_are_known = lambda *args, **kwargs: None
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        else:
            train_step = self._train_step

        scheduler = kwargs.get("scheduler")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        train_ds = tf.data.Dataset.from_tensor_slices((x_preprocessed, y_preprocessed)).shuffle(10000).batch(batch_size)

        for epoch in tqdm(range(nb_epochs), disable=not verbose, desc="Epochs"):
            for images, labels in train_ds:
                train_step(self.model, images, labels)

            if scheduler is not None:
                scheduler(epoch)

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, verbose: bool = False, **kwargs) -> None:
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in TensorFlow, it will.
        :param nb_epochs: Number of epochs to use for training.
        :param verbose: Display training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. This parameter currently supports
                       "scheduler" which is an optional function that will be called at the end of every
                       epoch to adjust the learning rate.
        """
        import tensorflow as tf
        from art.data_generators import TensorFlowV2DataGenerator

        if self._train_step is None:  # pragma: no cover
            if self._loss_object is None:  # pragma: no cover
                raise TypeError(
                    "A loss function `loss_object` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )
            if self._optimizer is None:  # pragma: no cover
                raise ValueError(
                    "An optimizer `optimizer` or training function `train_step` is required for fitting the "
                    "model, but it has not been defined."
                )

            @tf.function
            def train_step(model, images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = self.loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        else:
            train_step = self._train_step

        scheduler = kwargs.get("scheduler")

        # Train directly in TensorFlow
        from art.preprocessing.standardisation_mean_std.tensorflow import StandardisationMeanStdTensorFlow

        if isinstance(generator, TensorFlowV2DataGenerator) and (
            self.preprocessing is None
            or (
                isinstance(self.preprocessing, StandardisationMeanStdTensorFlow)
                and (
                    self.preprocessing.mean,
                    self.preprocessing.std,
                )
                == (0, 1)
            )
        ):
            for epoch in tqdm(range(nb_epochs), disable=not verbose, desc="Epochs"):
                for i_batch, o_batch in generator.iterator:
                    if self._reduce_labels:
                        o_batch = tf.math.argmax(o_batch, axis=1)
                    train_step(self._model, i_batch, o_batch)

                if scheduler is not None:
                    scheduler(epoch)
        else:
            # Fit a generic data generator through the API
            super().fit_generator(generator, nb_epochs=nb_epochs)

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
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        import tensorflow as tf

        x = tf.convert_to_tensor(x)

        with tf.GradientTape(persistent=True) as tape:
            # Apply preprocessing
            if self.all_framework_preprocessing:
                x_grad = tf.convert_to_tensor(x)
                tape.watch(x_grad)
                x_input, _ = self._apply_preprocessing(x_grad, y=None, fit=False)
            else:
                x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
                x_grad = tf.convert_to_tensor(x_preprocessed)
                tape.watch(x_grad)
                x_input = x_grad

            tape.watch(x_input)

            # Compute the gradients
            if tf.executing_eagerly():
                if label is None:
                    # Compute the gradients w.r.t. all classes
                    class_gradients = []

                    for i in range(self.nb_classes):
                        predictions = self.model(x_input, training=training_mode)
                        prediction = predictions[:, i]
                        tape.watch(prediction)

                        class_gradient = tape.gradient(prediction, x_input).numpy()
                        class_gradients.append(class_gradient)
                        # Break after 1 iteration for binary classification case
                        if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                            break

                    gradients = np.swapaxes(np.array(class_gradients), 0, 1)

                elif isinstance(label, int):
                    # Compute the gradients only w.r.t. the provided label
                    predictions = self.model(x_input, training=training_mode)
                    prediction = predictions[:, label]
                    tape.watch(prediction)

                    class_gradient = tape.gradient(prediction, x_grad).numpy()
                    gradients = np.expand_dims(class_gradient, axis=1)

                else:
                    # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
                    class_gradients = []
                    unique_labels = list(np.unique(label))

                    for unique_label in unique_labels:
                        predictions = self.model(x_input, training=training_mode)
                        prediction = predictions[:, unique_label]
                        tape.watch(prediction)

                        class_gradient = tape.gradient(prediction, x_grad).numpy()
                        class_gradients.append(class_gradient)

                    gradients = np.swapaxes(np.array(class_gradients), 0, 1)
                    lst = [unique_labels.index(i) for i in label]
                    gradients = np.expand_dims(gradients[np.arange(len(gradients)), lst], axis=1)

                if not self.all_framework_preprocessing:
                    gradients = self._apply_preprocessing_gradient(x, gradients)

            else:
                raise NotImplementedError("Expecting eager execution.")

        return gradients

    def compute_loss(
        self,
        x: np.ndarray | "tf.Tensor",
        y: np.ndarray | "tf.Tensor",
        reduction: str = "none",
        training_mode: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute the loss.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of losses of the same shape as `x`.
        """
        import tensorflow as tf
        from art.utils import check_and_transform_label_format

        if self._loss_object is None:  # pragma: no cover
            raise TypeError("The loss function `loss_object` is required for computing losses, but it is not defined.")
        prev_reduction = self._loss_object.reduction
        if reduction == "none":
            self._loss_object.reduction = tf.keras.losses.Reduction.NONE
        elif reduction == "mean":
            self._loss_object.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        elif reduction == "sum":
            self._loss_object.reduction = tf.keras.losses.Reduction.SUM

        y = check_and_transform_label_format(y, self.nb_classes)  # type: ignore

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        if tf.executing_eagerly():
            x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)
            predictions = self.model(x_preprocessed_tf, training=training_mode)
            if self._reduce_labels:
                loss = self._loss_object(np.argmax(y_preprocessed, axis=1), predictions)
            else:
                loss = self._loss_object(y_preprocessed, predictions)
        else:
            raise NotImplementedError("Expecting eager execution.")

        self._loss_object.reduction = prev_reduction
        return loss.numpy()

    def compute_losses(
        self,
        x: np.ndarray | "tf.Tensor",
        y: np.ndarray | "tf.Tensor",
        reduction: str = "none",
    ) -> dict[str, np.ndarray | "tf.Tensor"]:
        """
        Compute all loss components.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :return: Dictionary of loss components.
        """
        return {"total": self.compute_loss(x=x, y=y, reduction=reduction)}

    def loss_gradient(
        self,
        x: np.ndarray | "tf.Tensor",
        y: np.ndarray | "tf.Tensor",
        training_mode: bool = False,
        **kwargs,
    ) -> np.ndarray | "tf.Tensor":
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        import tensorflow as tf

        if self._loss_object is None:  # pragma: no cover
            raise TypeError(
                "The loss function `loss_object` is required for computing loss gradients, but it has not been "
                "defined."
            )

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                # Apply preprocessing
                if self.all_framework_preprocessing:
                    x_grad = tf.convert_to_tensor(x)
                    tape.watch(x_grad)
                    x_input, y_input = self._apply_preprocessing(x_grad, y=y, fit=False)
                else:
                    x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False)
                    x_grad = tf.convert_to_tensor(x_preprocessed)
                    tape.watch(x_grad)
                    x_input = x_grad
                    y_input = y_preprocessed

                predictions = self.model(x_input, training=training_mode)

                if self._reduce_labels:
                    loss = self._loss_object(np.argmax(y_input, axis=1), predictions)
                else:
                    loss = self._loss_object(y_input, predictions)

            gradients = tape.gradient(loss, x_grad)

            if isinstance(x, np.ndarray):
                gradients = gradients.numpy()

        else:
            raise NotImplementedError("Expecting eager execution.")

        # Apply preprocessing gradients
        if not self.all_framework_preprocessing:
            gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def clone_for_refitting(
        self,
    ) -> "TensorFlowV2Classifier":  # pragma: no cover
        """
        Create a copy of the classifier that can be refit from scratch. Will inherit same architecture, optimizer and
        initialization as cloned model, but without weights.

        :return: new estimator
        """
        import tensorflow as tf

        try:
            model = tf.keras.models.clone_model(self.model, input_tensors=self.model.input)
        except ValueError as error:
            raise ValueError("Cannot clone custom tensorflow models") from error

        # Recreate optimizer from config (fresh instance for each clone)
        optimizer_config = self.model.optimizer.get_config()
        optimizer_class = type(self.model.optimizer)
        new_optimizer = optimizer_class.from_config(optimizer_config)

        # Compile the model with the new optimizer
        model.compile(
            optimizer=new_optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics,
            run_eagerly=self.model.run_eagerly,
        )

        clone = type(self)(model, self.nb_classes, self.input_shape)
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        clone._train_step = self._train_step
        clone._reduce_labels = self._reduce_labels
        clone._loss_object = self._loss_object
        clone._optimizer = new_optimizer
        return clone

    def reset(self) -> None:
        """
        Resets the weights of the classifier so that it can be refit from scratch.
        """
        import tensorflow as tf

        for layer in self.model.layers:
            if isinstance(layer, (tf.keras.Model, tf.keras.models.Sequential)):  # if there is a model as a layer
                raise NotImplementedError("Resetting of models with models as layers has not been tested.")
            #     self.reset(layer)  # apply recursively
            #     continue

            # find initializers
            if hasattr(layer, "cell"):
                init_container = layer.cell
            else:
                init_container = layer

            for key, initializer in init_container.__dict__.items():
                if "initializer" not in key:  # not an initializer skip
                    continue

                # find the corresponding variable, like the kernel or the bias
                if key == "recurrent_initializer":  # special case check
                    var = getattr(init_container, "recurrent_kernel", None)
                else:
                    var = getattr(init_container, key.replace("_initializer", ""), None)

                if var is not None:
                    var.assign(initializer(var.shape, var.dtype))

    def _get_layers(self) -> list:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        """
        raise NotImplementedError

    @property
    def layer_names(self) -> list[str] | None:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        import tensorflow as tf

        if isinstance(self._model, (tf.keras.Model, tf.keras.models.Sequential)):
            return [layer.name for layer in self._model.layers if hasattr(layer, "name")]

        return None  # type: ignore

    def get_activations(  # type: ignore
        self, x: np.ndarray, layer: int | str, batch_size: int = 128, framework: bool = False
    ) -> np.ndarray | None:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations.
        :param batch_size: Batch size.
        :param framework: Return activation as tensor.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        import tensorflow as tf
        import tensorflow.keras.backend as k
        from art.config import ART_NUMPY_DTYPE

        if not isinstance(self._model, tf.keras.models.Sequential):  # pragma: no cover
            raise ValueError("Method get_activations is not supported for non-Sequential models.")

        i_layer = None
        if self.layer_names is None:  # pragma: no cover
            raise ValueError("No layer names identified.")

        if isinstance(layer, six.string_types):
            if layer not in self.layer_names:  # pragma: no cover
                raise ValueError(f"Layer name {layer} is not part of the graph.")
            for i_name, name in enumerate(self.layer_names):
                if name == layer:
                    i_layer = i_name
                    break
        elif isinstance(layer, int):
            if layer < -len(self.layer_names) or layer >= len(self.layer_names):  # pragma: no cover
                raise ValueError(
                    f"Layer index {layer} is outside of range (-{len(self.layer_names)} "
                    f"to {len(self.layer_names) - 1})."
                )
            i_layer = layer
        else:  # pragma: no cover
            raise TypeError("Layer must be of type `str` or `int`.")

        activation_model = tf.keras.Model(self._model.layers[0].input, self._model.layers[i_layer].output)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False)

        if framework:
            if isinstance(x_preprocessed, tf.Tensor):
                return activation_model(x_preprocessed, training=False)
            return activation_model(tf.convert_to_tensor(x_preprocessed), training=False)

        # Determine shape of expected output and prepare array
        layer_instance = self._model.layers[i_layer]
        input_shape = k.int_shape(layer_instance.input)  # Keras 3.x-safe way
        output_shape = layer_instance.compute_output_shape(input_shape)
        activations = np.zeros((x_preprocessed.shape[0],) + output_shape[1:], dtype=ART_NUMPY_DTYPE)

        # Get activations with batching
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = (
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, x_preprocessed.shape[0]),
            )
            activations[begin:end] = activation_model([x_preprocessed[begin:end]], training=False).numpy()

        return activations

    def save(self, filename: str, path: str | None = None) -> None:
        """
        Save a model to file in the format specific to the backend framework. For TensorFlow, .ckpt is used.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        raise NotImplementedError

    def __repr__(self):
        repr_ = (
            f"{self.__module__ + '.' + self.__class__.__name__}(model={self._model}, nb_classes={self.nb_classes}, "
            f"input_shape={self._input_shape}, loss_object={self._loss_object}, optimizer={self.optimizer}, "
            f"train_step={self._train_step}, channels_first={self.channels_first}, clip_values={self.clip_values!r}, "
            f"preprocessing_defences={self.preprocessing_defences}, "
            f"postprocessing_defences={self.postprocessing_defences}, preprocessing={self.preprocessing})"
        )

        return repr_
