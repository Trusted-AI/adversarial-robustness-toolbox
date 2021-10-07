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
This module implements the classifier `MXClassifier` for MXNet Gluon models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import six

from art import config
from art.estimators.mxnet import MXEstimator
from art.estimators.classification.classifier import ClassGradientsMixin, ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import mxnet as mx

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class MXClassifier(ClassGradientsMixin, ClassifierMixin, MXEstimator):  # lgtm [py/missing-call-to-init]
    """
    Class for importing MXNet Gluon models.
    """

    estimator_params = (
        MXEstimator.estimator_params
        + ClassifierMixin.estimator_params
        + [
            "loss",
            "input_shape",
            "nb_classes",
            "optimizer",
            "ctx",
            "channels_first",
        ]
    )

    def __init__(
        self,
        model: "mx.gluon.Block",
        loss: Union["mx.nd.loss", "mx.gluon.loss"],
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["mx.gluon.Trainer"] = None,
        ctx: Optional["mx.context.Context"] = None,
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Initialize an `MXClassifier` object. Assumes the `model` passed as parameter is a Gluon model.

        :param model: The Gluon model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param optimizer: The optimizer used to train the classifier. This parameter is only required if fitting will
                          be done with method fit.
        :param ctx: The device on which the model runs (CPU or GPU). If not provided, CPU is assumed.
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
        import mxnet as mx  # lgtm [py/repeated-import]

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._loss = loss
        self._nb_classes = nb_classes
        self._input_shape = input_shape
        self._device = ctx
        self._optimizer = optimizer

        if ctx is None:
            self._ctx = mx.cpu()
        else:
            self._ctx = ctx

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
    def loss(self) -> Union["mx.nd.loss", "mx.gluon.loss"]:
        """
        Return the loss function.

        :return: The loss function.
        """
        return self._loss  # type: ignore

    @property
    def optimizer(self) -> "mx.gluon.Trainer":
        """
        Return the optimizer used to train the classifier.

        :return: The optimizer used to train the classifier.
        """
        return self._optimizer  # type: ignore

    @property
    def ctx(self) -> "mx.context.Context":
        """
        Return the device on which the model runs.

        :return: The device on which the model runs (CPU or GPU).
        """
        return self._ctx  # type: ignore

    def fit(
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs
    ) -> None:  # pragma: no cover
        """
        Fit the classifier on the training set `(inputs, outputs)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for MXNet
               and providing it takes no effect.
        """
        import mxnet as mx  # lgtm [py/repeated-import]

        if self.optimizer is None:
            raise ValueError("An MXNet optimizer is required for fitting the model.")

        training_mode = True

        y = check_and_transform_label_format(y, self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)
        y_preprocessed = np.argmax(y_preprocessed, axis=1)
        nb_batch = int(np.ceil(len(x_preprocessed) / batch_size))
        ind = np.arange(len(x_preprocessed))

        for _ in range(nb_epochs):
            # Shuffle the examples
            np.random.shuffle(ind)

            # Train for one epoch
            for m in range(nb_batch):
                x_batch = mx.nd.array(
                    x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]].astype(config.ART_NUMPY_DTYPE)
                ).as_in_context(self.ctx)
                y_batch = mx.nd.array(y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]).as_in_context(
                    self.ctx
                )

                with mx.autograd.record(train_mode=training_mode):
                    # Perform prediction
                    preds = self._model(x_batch)

                    # Apply postprocessing
                    preds = self._apply_postprocessing(preds=preds, fit=True)

                    # Form the loss function
                    loss = self.loss(preds, y_batch)

                loss.backward()

                # Update parameters
                self.optimizer.step(batch_size)

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:  # pragma: no cover
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for MXNet
               and providing it takes no effect.
        """
        import mxnet as mx  # lgtm [py/repeated-import]
        from art.data_generators import MXDataGenerator

        if self.optimizer is None:
            raise ValueError("An MXNet optimizer is required for fitting the model.")

        training_mode = True

        if (
            isinstance(generator, MXDataGenerator)
            and (self.preprocessing is None or self.preprocessing == [])
            and self.preprocessing == (0, 1)
        ):
            # Train directly in MXNet
            for _ in range(nb_epochs):
                for x_batch, y_batch in generator.iterator:
                    x_batch = mx.nd.array(x_batch.astype(config.ART_NUMPY_DTYPE)).as_in_context(self.ctx)
                    y_batch = mx.nd.argmax(y_batch, axis=1)
                    y_batch = mx.nd.array(y_batch).as_in_context(self.ctx)

                    with mx.autograd.record(train_mode=training_mode):
                        # Perform prediction
                        preds = self._model(x_batch)

                        # Form the loss function
                        loss = self.loss(preds, y_batch)

                    loss.backward()

                    # Update parameters
                    self.optimizer.step(x_batch.shape[0])
        else:
            # Fit a generic data generator through the API
            super().fit_generator(generator, nb_epochs=nb_epochs)

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
        import mxnet as mx  # lgtm [py/repeated-import]

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Predict
            x_batch = mx.nd.array(x_preprocessed[begin:end].astype(config.ART_NUMPY_DTYPE), ctx=self.ctx)
            x_batch.attach_grad()
            with mx.autograd.record(train_mode=training_mode):
                preds = self._model(x_batch)

            results[begin:end] = preds.asnumpy()

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

    def class_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, label: Union[int, List[int], None] = None, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        import mxnet as mx  # lgtm [py/repeated-import]

        # Check value of label for computing gradients
        if not (  # pragma: no cover
            label is None
            or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self.nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % str(label))

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = mx.nd.array(x_preprocessed.astype(config.ART_NUMPY_DTYPE), ctx=self.ctx)
        x_preprocessed.attach_grad()

        if label is None:
            with mx.autograd.record(train_mode=False):
                preds = self._model(x_preprocessed)
                class_slices = [preds[:, i] for i in range(self.nb_classes)]

            grads = []
            for slice_ in class_slices:
                slice_.backward(retain_graph=True)
                grad = x_preprocessed.grad.asnumpy()
                grads.append(grad)
            grads = np.swapaxes(np.array(grads), 0, 1)
        elif isinstance(label, (int, np.integer)):
            with mx.autograd.record(train_mode=training_mode):
                preds = self._model(x_preprocessed)
                class_slice = preds[:, label]

            class_slice.backward()
            grads = np.expand_dims(x_preprocessed.grad.asnumpy(), axis=1)
        else:
            unique_labels = list(np.unique(label))

            with mx.autograd.record(train_mode=training_mode):
                preds = self._model(x_preprocessed)
                class_slices = [preds[:, i] for i in unique_labels]

            grads = []
            for slice_ in class_slices:
                slice_.backward(retain_graph=True)
                grad = x_preprocessed.grad.asnumpy()
                grads.append(grad)

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_labels.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]
            grads = np.expand_dims(grads, axis=1)

        grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    def loss_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        import mxnet as mx  # lgtm [py/repeated-import]

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)
        y_preprocessed = mx.nd.array([np.argmax(y_preprocessed, axis=1)], ctx=self.ctx).T
        x_preprocessed = mx.nd.array(x_preprocessed.astype(config.ART_NUMPY_DTYPE), ctx=self.ctx)
        x_preprocessed.attach_grad()

        with mx.autograd.record(train_mode=training_mode):
            preds = self._model(x_preprocessed)
            loss = self.loss(preds, y_preprocessed)

        loss.backward()

        # Compute gradients
        grads = x_preprocessed.grad.asnumpy()
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    @property
    def layer_names(self) -> List[str]:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        return self._layer_names

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int = 128, framework: bool = False
    ) -> np.ndarray:  # pragma: no cover
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations
        :param batch_size: Size of batches.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        import mxnet as mx  # lgtm [py/repeated-import]

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError("Layer name %s is not part of the model." % layer)
            layer_ind = self._layer_names.index(layer)
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError(
                    "Layer index %d is outside of range (0 to %d included)." % (layer, len(self._layer_names) - 1)
                )
            layer_ind = layer
        else:
            raise TypeError("Layer must be of type `str` or `int`.")

        # Apply preprocessing and defences
        if x.shape == self.input_shape:
            x_expanded = np.expand_dims(x, 0)
        else:
            x_expanded = x

        x_preprocessed, _ = self._apply_preprocessing(x=x_expanded, y=None, fit=False)

        if framework:
            return self._model[layer_ind]

        # Compute activations with batching
        activations = []
        nb_batches = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for batch_index in range(nb_batches):
            # Batch indexes
            begin, end = (
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Predict
            x_batch = mx.nd.array(x_preprocessed[begin:end].astype(config.ART_NUMPY_DTYPE), ctx=self.ctx)
            x_batch.attach_grad()
            with mx.autograd.record(train_mode=False):
                preds = self._model[layer_ind](x_batch)

            activations.append(preds.asnumpy())

        activations = np.vstack(activations)
        return activations

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework. For Gluon, only parameters are saved in
        file with name `<filename>.params` at the specified path. To load the saved model, the original model code needs
        to be run before calling `load_parameters` on the generated Gluon model.

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

        self._model.save_parameters(full_path + ".params")
        logger.info("Model parameters saved in path: %s.params.", full_path)

    def __repr__(self):
        repr_ = (
            "%s(model=%r, loss=%r, input_shape=%r, nb_classes=%r, optimizer=%r, ctx=%r, "
            " channels_first=%r, clip_values=%r, preprocessing=%r, postprocessing_defences=%r,"
            " preprocessing=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._model,
                self.loss,
                self.input_shape,
                self.nb_classes,
                self.optimizer,
                self.ctx,
                self.channels_first,
                self.clip_values,
                self.preprocessing,
                self.postprocessing_defences,
                self.preprocessing,
            )
        )

        return repr_

    def _get_layers(self) -> list:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        """
        import mxnet

        if isinstance(self._model, mxnet.gluon.nn.Sequential):
            layer_names = [layer.name for layer in self._model[:-1]]
            logger.info("Inferred %i hidden layers on MXNet classifier.", len(layer_names))
        else:
            layer_names = []

        return layer_names
