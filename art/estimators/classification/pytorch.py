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
This module implements the classifier `PyTorchClassifier` for PyTorch models.
"""
# pylint: disable=C0302
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import six

from art import config
from art.estimators.classification.classifier import (
    ClassGradientsMixin,
    ClassifierMixin,
)
from art.estimators.pytorch import PyTorchEstimator
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412, C0302
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchClassifier(ClassGradientsMixin, ClassifierMixin, PyTorchEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements a classifier with the PyTorch framework.
    """

    estimator_params = (
        PyTorchEstimator.estimator_params
        + ClassifierMixin.estimator_params
        + [
            "loss",
            "input_shape",
            "optimizer",
            "use_amp",
            "opt_level",
            "loss_scale",
        ]
    )

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        use_amp: bool = False,
        opt_level: str = "O1",
        loss_scale: Optional[Union[float, str]] = "dynamic",
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
    ) -> None:
        """
        Initialization specifically for the PyTorch-based implementation.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param optimizer: The optimizer used to train the classifier.
        :param use_amp: Whether to use the automatic mixed precision tool to enable mixed precision training or
                        gradient computation, e.g. with loss gradient computation. When set to True, this option is
                        only triggered if there are GPUs available.
        :param opt_level: Specify a pure or mixed precision optimization level. Used when use_amp is True. Accepted
                          values are `O0`, `O1`, `O2`, and `O3`.
        :param loss_scale: Loss scaling. Used when use_amp is True. If passed as a string, must be a string
                           representing a number, e.g., “1.0”, or the string “dynamic”.
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
        """
        import torch  # lgtm [py/repeated-import]

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )
        self._nb_classes = nb_classes
        self._input_shape = input_shape
        self._model = self._make_model_wrapper(model)
        self._loss = loss
        self._optimizer = optimizer
        self._use_amp = use_amp
        self._learning_phase: Optional[bool] = None
        self._opt_level = opt_level
        self._loss_scale = loss_scale

        # Check if model is RNN-like to decide if freezing batch-norm and dropout layers might be required for loss and
        # class gradient calculation
        self.is_rnn = any((isinstance(m, torch.nn.modules.RNNBase) for m in self._model.modules()))

        # Get the internal layers
        self._layer_names: List[str] = self._model.get_layers  # type: ignore

        self._model.to(self._device)

        # Index of layer at which the class gradients should be calculated
        self._layer_idx_gradients = -1

        if isinstance(
            self._loss,
            (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss),
        ):
            self._reduce_labels = True
            self._int_labels = True
        elif isinstance(
            self._loss,
            (torch.nn.BCELoss),
        ):
            self._reduce_labels = True
            self._int_labels = False
        else:
            self._reduce_labels = False
            self._int_labels = False

        # Setup for AMP use
        if self._use_amp:
            from apex import amp  # pylint: disable=E0611

            if self._optimizer is None:
                logger.warning(
                    "An optimizer is needed to use the automatic mixed precision tool, but none for provided. "
                    "A default optimizer is used."
                )

                # Create the optimizers
                parameters = self._model.parameters()
                self._optimizer = torch.optim.SGD(parameters, lr=0.01)

            if self.device.type == "cpu":
                enabled = False
            else:
                enabled = True

            self._model, self._optimizer = amp.initialize(
                models=self._model,
                optimizers=self._optimizer,
                enabled=enabled,
                opt_level=opt_level,
                loss_scale=loss_scale,
            )

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    @property
    def model(self) -> "torch.nn.Module":
        return self._model._model  # pylint: disable=W0212

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def loss(self) -> "torch.nn.modules.loss._Loss":
        """
        Return the loss function.

        :return: The loss function.
        """
        return self._loss  # type: ignore

    @property
    def optimizer(self) -> "torch.optim.Optimizer":
        """
        Return the optimizer.

        :return: The optimizer.
        """
        return self._optimizer  # type: ignore

    @property
    def use_amp(self) -> bool:
        """
        Return a boolean indicating whether to use the automatic mixed precision tool.

        :return: Whether to use the automatic mixed precision tool.
        """
        return self._use_amp  # type: ignore

    @property
    def opt_level(self) -> str:
        """
        Return a string specifying a pure or mixed precision optimization level.

        :return: A string specifying a pure or mixed precision optimization level. Possible
                 values are `O0`, `O1`, `O2`, and `O3`.
        """
        return self._opt_level  # type: ignore

    @property
    def loss_scale(self) -> Union[float, str]:
        """
        Return the loss scaling value.

        :return: Loss scaling. Possible values for string: a string representing a number, e.g., “1.0”,
                 or the string “dynamic”.
        """
        return self._loss_scale  # type: ignore

    def reduce_labels(self, y: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Reduce labels from one-hot encoded to index labels.
        """
        # pylint: disable=R0911
        import torch  # lgtm [py/repeated-import]

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        # Checking for exactly 2 classes to support binary classification
        if self.nb_classes > 2 or (self.nb_classes == 2 and len(y.shape) == 2 and y.shape[1] == 2):
            if self._reduce_labels and self._int_labels:
                if isinstance(y, torch.Tensor):
                    return torch.argmax(y, dim=1)
                return np.argmax(y, axis=1)
            if self._reduce_labels:  # float labels
                if isinstance(y, torch.Tensor):
                    return torch.argmax(y, dim=1).type("torch.FloatTensor")
                y_index = np.argmax(y, axis=1).astype(np.float32)
                y_index = np.expand_dims(y_index, axis=1)
                return y_index
            return y

        if isinstance(y, torch.Tensor):
            return y.float()
        return y.astype(np.float32)

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
        import torch  # lgtm [py/repeated-import]

        # Set model mode
        self._model.train(mode=training_mode)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        results_list = []

        # Run prediction with batch processing
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            with torch.no_grad():
                model_outputs = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            output = model_outputs[-1]
            output = output.detach().cpu().numpy().astype(np.float32)
            if len(output.shape) == 1:
                output = np.expand_dims(output.detach().cpu().numpy(), axis=1).astype(np.float32)

            results_list.append(output)

        results = np.vstack(results_list)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

    def _predict_framework(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :return: Tensor of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False, no_grad=False)

        # Put the model in the eval mode
        self._model.eval()

        model_outputs = self._model(x_preprocessed)
        output = model_outputs[-1]

        return output

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch  # lgtm [py/repeated-import]

        # Put the model in the training mode
        self._model.train()

        if self._optimizer is None:
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = torch.from_numpy(x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self._model(i_batch)

                # Form the loss function
                loss = self._loss(model_outputs[-1], o_batch)  # lgtm [py/call-to-non-callable]

                # Do training
                if self._use_amp:
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self._optimizer.step()

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch  # lgtm [py/repeated-import]
        from art.data_generators import PyTorchDataGenerator

        # Put the model in the training mode
        self._model.train()

        if self._optimizer is None:
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        # Train directly in PyTorch
        if isinstance(generator, PyTorchDataGenerator) and not self.preprocessing:
            for _ in range(nb_epochs):
                for i_batch, o_batch in generator.iterator:
                    if isinstance(i_batch, np.ndarray):
                        i_batch = torch.from_numpy(i_batch).to(self._device)
                    else:
                        i_batch = i_batch.to(self._device)

                    if isinstance(o_batch, np.ndarray):
                        o_batch = torch.argmax(torch.from_numpy(o_batch).to(self._device), dim=1)
                    else:
                        o_batch = torch.argmax(o_batch.to(self._device), dim=1)

                    # Zero the parameter gradients
                    self._optimizer.zero_grad()

                    # Perform prediction
                    model_outputs = self._model(i_batch)

                    # Form the loss function
                    loss = self._loss(model_outputs[-1], o_batch)

                    # Do training
                    if self._use_amp:
                        from apex import amp  # pylint: disable=E0611

                        with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                            scaled_loss.backward()

                    else:
                        loss.backward()

                    self._optimizer.step()

        else:
            # Fit a generic data generator through the API
            super().fit_generator(generator, nb_epochs=nb_epochs)

    def clone_for_refitting(self) -> "PyTorchClassifier":  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Create a copy of the classifier that can be refit from scratch. Will inherit same architecture, optimizer and
        initialization as cloned model, but without weights.

        :return: new estimator
        """
        model = copy.deepcopy(self.model)
        clone = type(self)(model, self._loss, self.input_shape, self.nb_classes, optimizer=self._optimizer)
        # reset weights
        clone.reset()
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        return clone

    def reset(self) -> None:
        """
        Resets the weights of the classifier so that it can be refit from scratch.

        """

        def weight_reset(module):
            reset_parameters = getattr(module, "reset_parameters", None)
            if reset_parameters and callable(reset_parameters):
                module.reset_parameters()

        self.model.apply(weight_reset)

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
                              Note on RNN-like models: Backpropagation through RNN modules in eval mode raises
                              RuntimeError due to cudnn issues and require training mode, i.e. RuntimeError: cudnn RNN
                              backward can only be called in training mode. Therefore, if the model is an RNN type we
                              always use training mode but freeze batch-norm and dropout layers if
                              `training_mode=False.`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        import torch  # lgtm [py/repeated-import]

        self._model.train(mode=training_mode)

        # Backpropagation through RNN modules in eval mode raises RuntimeError due to cudnn issues and require training
        # mode, i.e. RuntimeError: cudnn RNN backward can only be called in training mode. Therefore, if the model is
        # an RNN type we always use training mode but freeze batch-norm and dropout layers if training_mode=False.
        if self.is_rnn:
            self._model.train(mode=True)
            if not training_mode:
                logger.debug(
                    "Freezing batch-norm and dropout layers for gradient calculation in train mode with eval parameters"
                    "of batch-norm and dropout."
                )
                self.set_batchnorm(train=False)
                self.set_dropout(train=False)

        if not (
            (label is None)
            or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self._nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Apply preprocessing
        if self.all_framework_preprocessing:
            x_grad = torch.from_numpy(x).to(self._device)
            if self._layer_idx_gradients < 0:
                x_grad.requires_grad = True
            x_input, _ = self._apply_preprocessing(x_grad, y=None, fit=False, no_grad=False)
        else:
            x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            if self._layer_idx_gradients < 0:
                x_grad.requires_grad = True
            x_input = x_grad

        # Run prediction
        model_outputs = self._model(x_input)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_grad

        # Set where to get gradient from
        preds = model_outputs[-1]

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self._model.zero_grad()
        if label is None:
            if len(preds.shape) == 1 or preds.shape[1] == 1:
                num_outputs = 1
            else:
                num_outputs = self.nb_classes

            for i in range(num_outputs):
                torch.autograd.backward(
                    preds[:, i],
                    torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                    retain_graph=True,
                )

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label],
                torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                retain_graph=True,
            )
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(
                    preds[:, i],
                    torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                    retain_graph=True,
                )

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    def compute_loss(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        reduction: str = "none",
        **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :return: Array of losses of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        self._model.eval()

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        if isinstance(x, torch.Tensor):
            inputs_t = x_preprocessed
            labels_t = y_preprocessed
        else:
            # Convert the inputs to Tensors
            inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
            # Convert the labels to Tensors
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the loss and return
        model_outputs = self._model(inputs_t)
        prev_reduction = self._loss.reduction

        # Return individual loss values
        self._loss.reduction = reduction
        loss = self._loss(model_outputs[-1], labels_t)
        self._loss.reduction = prev_reduction

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()

    def compute_losses(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        reduction: str = "none",
    ) -> Dict[str, Union[np.ndarray, "torch.Tensor"]]:
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

    def loss_gradient(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        training_mode: bool = False,
        **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
                              Note on RNN-like models: Backpropagation through RNN modules in eval mode raises
                              RuntimeError due to cudnn issues and require training mode, i.e. RuntimeError: cudnn RNN
                              backward can only be called in training mode. Therefore, if the model is an RNN type we
                              always use training mode but freeze batch-norm and dropout layers if
                              `training_mode=False.`
        :return: Array of gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        self._model.train(mode=training_mode)

        # Backpropagation through RNN modules in eval mode raises RuntimeError due to cudnn issues and require training
        # mode, i.e. RuntimeError: cudnn RNN backward can only be called in training mode. Therefore, if the model is
        # an RNN type we always use training mode but freeze batch-norm and dropout layers if training_mode=False.
        if self.is_rnn:
            self._model.train(mode=True)
            if not training_mode:
                logger.debug(
                    "Freezing batch-norm and dropout layers for gradient calculation in train mode with eval parameters"
                    "of batch-norm and dropout."
                )
                self.set_batchnorm(train=False)
                self.set_dropout(train=False)

        # Apply preprocessing
        if self.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                x_grad = x.clone().detach().requires_grad_(True)
            else:
                x_grad = torch.tensor(x).to(self._device)
                x_grad.requires_grad = True
            if isinstance(y, torch.Tensor):
                y_grad = y.clone().detach()
            else:
                y_grad = torch.tensor(y).to(self._device)
            inputs_t, y_preprocessed = self._apply_preprocessing(x_grad, y=y_grad, fit=False, no_grad=False)
        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            x_grad.requires_grad = True
            inputs_t = x_grad
        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)
        else:
            labels_t = y_preprocessed

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        loss = self._loss(model_outputs[-1], labels_t)  # lgtm [py/call-to-non-callable]

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        if self._use_amp:
            from apex import amp  # pylint: disable=E0611

            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            loss.backward()

        if isinstance(x, torch.Tensor):
            grads = x_grad.grad
        else:
            grads = x_grad.grad.cpu().numpy().copy()  # type: ignore

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

    def get_activations(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        layer: Optional[Union[int, str]] = None,
        batch_size: int = 128,
        framework: bool = False,
    ) -> np.ndarray:
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
        import torch  # lgtm [py/repeated-import]

        self._model.eval()

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False)

        # Get index of the extracted layer
        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError("Layer name %s not supported" % layer)
            layer_index = self._layer_names.index(layer)

        elif isinstance(layer, (int, np.integer)):
            layer_index = layer

        else:
            raise TypeError("Layer must be of type str or int")

        if framework:
            if isinstance(x, torch.Tensor):
                return self._model(x)[layer_index]
            return self._model(torch.from_numpy(x).to(self._device))[layer_index]

        # Run prediction with batch processing
        results = []
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Run prediction for the current batch
            layer_output = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))[layer_index]
            results.append(layer_output.detach().cpu().numpy())

        results = np.concatenate(results)

        return results

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        import torch  # lgtm [py/repeated-import]

        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        # pylint: disable=W0212
        # disable pylint because access to _modules required
        torch.save(self._model._model.state_dict(), full_path + ".model")
        torch.save(self._optimizer.state_dict(), full_path + ".optimizer")  # type: ignore
        logger.info("Model state dict saved in path: %s.", full_path + ".model")
        logger.info("Optimizer state dict saved in path: %s.", full_path + ".optimizer")

    def __getstate__(self) -> Dict[str, Any]:
        """
        Use to ensure `PyTorchClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        """
        # pylint: disable=W0212
        # disable pylint because access to _model required
        state = self.__dict__.copy()
        state["inner_model"] = copy.copy(state["_model"]._model)

        # Remove the unpicklable entries
        del state["_model_wrapper"]
        del state["_device"]
        del state["_model"]

        model_name = str(time.time())
        state["model_name"] = model_name
        self.save(model_name)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Use to ensure `PyTorchClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        """
        import torch  # lgtm [py/repeated-import]

        # Recover model
        self.__dict__.update(state)
        full_path = os.path.join(config.ART_DATA_PATH, state["model_name"])
        model = state["inner_model"]
        model.load_state_dict(torch.load(str(full_path) + ".model"))
        model.eval()
        self._model = self._make_model_wrapper(model)

        # Recover device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Recover optimizer
        self._optimizer.load_state_dict(torch.load(str(full_path) + ".optimizer"))  # type: ignore

        self.__dict__.pop("model_name", None)
        self.__dict__.pop("inner_model", None)

    def __repr__(self):
        repr_ = (
            "%s(model=%r, loss=%r, optimizer=%r, input_shape=%r, nb_classes=%r, channels_first=%r, "
            "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._model,
                self._loss,
                self._optimizer,
                self._input_shape,
                self.nb_classes,
                self.channels_first,
                self.clip_values,
                self.preprocessing_defences,
                self.postprocessing_defences,
                self.preprocessing,
            )
        )

        return repr_

    def _make_model_wrapper(self, model: "torch.nn.Module") -> "torch.nn.Module":
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        try:
            import torch.nn as nn

            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(nn.Module):
                    """
                    This is a wrapper for the input model.
                    """

                    import torch  # lgtm [py/repeated-import]

                    def __init__(self, model: torch.nn.Module):
                        """
                        Initialization by storing the input model.

                        :param model: PyTorch model. The forward function of the model must return the logit output.
                        """
                        super().__init__()
                        self._model = model

                    # pylint: disable=W0221
                    # disable pylint because of API requirements for function
                    def forward(self, x):
                        """
                        This is where we get outputs from the input model.

                        :param x: Input data.
                        :type x: `torch.Tensor`
                        :return: a list of output layers, where the last 2 layers are logit and final outputs.
                        :rtype: `list`
                        """
                        # pylint: disable=W0212
                        # disable pylint because access to _model required
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
                                result.append(x)

                        elif isinstance(self._model, nn.Module):
                            x = self._model(x)
                            result.append(x)

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")

                        return result

                    @property
                    def get_layers(self) -> List[str]:
                        """
                        Return the hidden layers in the model, if applicable.

                        :return: The hidden layers in the model, input and output layers excluded.

                        .. warning:: `get_layers` tries to infer the internal structure of the model.
                                     This feature comes with no guarantees on the correctness of the result.
                                     The intended order of the layers tries to match their order in the model, but this
                                     is not guaranteed either. In addition, the function can only infer the internal
                                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                                     return the logit layer.
                        """
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            # pylint: disable=W0212
                            # disable pylint because access to _modules required
                            for name, module_ in self._model._modules.items():  # type: ignore
                                result.append(name + "_" + str(module_))

                        elif isinstance(self._model, nn.Module):
                            result.append("final_layer")

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")
                        logger.info(
                            "Inferred %i hidden layers on PyTorch classifier.",
                            len(result),
                        )

                        return result

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._model_wrapper(model)

        except ImportError:
            raise ImportError("Could not find PyTorch (`torch`) installation.") from ImportError
