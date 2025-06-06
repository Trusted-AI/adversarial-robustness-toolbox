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
This module implements the regressor `PyTorchRegressor` for PyTorch models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import copy
import logging
import os
import time
from typing import Any, TYPE_CHECKING

import numpy as np
import six

from art import config
from art.estimators.regression.regressor import RegressorMixin
from art.estimators.pytorch import PyTorchEstimator

if TYPE_CHECKING:

    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchRegressor(RegressorMixin, PyTorchEstimator):
    """
    This class implements a regressor with the PyTorch framework.
    """

    estimator_params = PyTorchEstimator.estimator_params + [
        "loss",
        "input_shape",
        "optimizer",
        "use_amp",
        "opt_level",
        "loss_scale",
    ]

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: tuple[int, ...],
        optimizer: "torch.optim.Optimizer" | None = None,  # type: ignore
        use_amp: bool = False,
        opt_level: str = "O1",
        loss_scale: float | str | None = "dynamic",
        channels_first: bool = True,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        preprocessing_defences: "Preprocessor" | list["Preprocessor"] | None = None,
        postprocessing_defences: "Postprocessor" | list["Postprocessor"] | None = None,
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
        :param optimizer: The optimizer used to train the regressor.
        :param use_amp: Whether to use the automatic mixed precision tool to enable mixed precision training or
                        gradient computation, e.g. with loss gradient computation. When set to True, this option is
                        only triggered if there are GPUs available.
        :param opt_level: Specify a pure or mixed precision optimization level. Used when use_amp is True. Accepted
                          values are `O0`, `O1`, `O2`, and `O3`.
        :param loss_scale: Loss scaling. Used when use_amp is True. If passed as a string, must be a string
                           representing a number, e.g., “1.0”, or the string “dynamic”.
        :param optimizer: The optimizer used to train the regressor.
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
        :param device_type: Type of device on which the regressor is run, either `gpu` or `cpu`.
        """
        import torch

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )
        self._input_shape = input_shape
        self._model = self._make_model_wrapper(model)
        self._loss = loss
        self._optimizer = optimizer
        self._use_amp = use_amp
        self._learning_phase: bool | None = None
        self._opt_level = opt_level
        self._loss_scale = loss_scale

        # Check if model is RNN-like to decide if freezing batch-norm and dropout layers might be required for loss and
        # class gradient calculation
        self.is_rnn = any((isinstance(m, torch.nn.modules.RNNBase) for m in self._model.modules()))

        # Get the internal layers
        self._layer_names: list[str] = self._model.get_layers  # type: ignore

        self._model.to(self._device)

        # Index of layer at which the class gradients should be calculated
        self._layer_idx_gradients = -1

        # Setup for AMP use
        if self._use_amp:  # pragma: no cover
            from apex import amp

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
        return self._model._model

    @property
    def input_shape(self) -> tuple[int, ...]:
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
    def loss_scale(self) -> float | str:
        """
        Return the loss scaling value.

        :return: Loss scaling. Possible values for string: a string representing a number, e.g., “1.0”,
                 or the string “dynamic”.
        """
        return self._loss_scale  # type: ignore

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Set model mode
        self._model.train(mode=training_mode)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Create dataloader
        x_tensor = torch.from_numpy(x_preprocessed)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        results_list = []
        for (x_batch,) in dataloader:
            # Move inputs to device
            x_batch = x_batch.to(self._device)

            # Run prediction
            with torch.no_grad():
                model_outputs = self._model(x_batch)
            output = model_outputs[-1]
            output = output.detach().cpu().numpy().astype(np.float32)
            if len(output.shape) == 1:
                output = np.expand_dims(output, axis=1).astype(np.float32)

            results_list.append(output)

        results = np.vstack(results_list)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

    def _predict_framework(
        self, x: "torch.Tensor", y: "torch.Tensor" | None = None
    ) -> tuple["torch.Tensor", "torch.Tensor" | None]:
        """
        Perform prediction for a batch of inputs.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :return: Tensor of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=False)

        # Put the model in the eval mode
        self._model.eval()

        model_outputs = self._model(x_preprocessed)
        output = model_outputs[-1]

        return output, y_preprocessed

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        drop_last: bool = False,
        scheduler: "torch.optim.lr_scheduler._LRScheduler" | None = None,
        **kwargs,
    ) -> None:
        """
        Fit the regressor on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Set model mode
        self._model.train(mode=training_mode)

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Create dataloader
        x_tensor = torch.from_numpy(x_preprocessed)
        y_tensor = torch.from_numpy(y_preprocessed)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        # Start training
        for _ in range(nb_epochs):
            # Train for one epoch
            for x_batch, y_batch in dataloader:
                # Move inputs to device
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self._model(x_batch)

                # Form the loss function
                loss = self._loss(model_outputs[-1].reshape(-1), y_batch)

                # Do training
                if self._use_amp:  # pragma: no cover
                    from apex import amp

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self._optimizer.step()

            if scheduler is not None:
                scheduler.step()

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the regressor using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch
        from art.data_generators import PyTorchDataGenerator

        # Put the model in the training mode
        self._model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        # Train directly in PyTorch
        from art.preprocessing.standardisation_mean_std.pytorch import StandardisationMeanStdPyTorch

        if isinstance(generator, PyTorchDataGenerator) and (
            self.preprocessing is None
            or (
                isinstance(self.preprocessing, StandardisationMeanStdPyTorch)
                and (
                    self.preprocessing.mean,
                    self.preprocessing.std,
                )
                == (0, 1)
            )
        ):
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
                    loss = self._loss(
                        model_outputs[-1].reshape(
                            -1,
                        ),
                        o_batch,
                    )

                    # Do training
                    if self._use_amp:  # pragma: no cover
                        from apex import amp

                        with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                            scaled_loss.backward()

                    else:
                        loss.backward()

                    self._optimizer.step()

        else:
            # Fit a generic data generator through the API
            super().fit_generator(generator, nb_epochs=nb_epochs)

    def clone_for_refitting(self) -> "PyTorchRegressor":
        """
        Create a copy of the regressor that can be refit from scratch. Will inherit same architecture, optimizer and
        initialization as cloned model, but without weights.

        :return: new estimator
        """
        model = copy.deepcopy(self.model)
        clone = type(self)(model, self._loss, self.input_shape, optimizer=self._optimizer)
        # reset weights
        clone.reset()
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        return clone

    def reset(self) -> None:
        """
        Resets the weights of the regressor so that it can be refit from scratch.

        """

        def weight_reset(module):
            reset_parameters = getattr(module, "reset_parameters", None)
            if reset_parameters and callable(reset_parameters):
                module.reset_parameters()

        self.model.apply(weight_reset)

    def compute_loss(  # type: ignore
        self,
        x: np.ndarray | "torch.Tensor",
        y: np.ndarray | "torch.Tensor",
        reduction: str = "none",
        **kwargs,
    ) -> np.ndarray | "torch.Tensor":
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
        import torch

        self._model.eval()

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        # y_preprocessed = self.reduce_labels(y_preprocessed)

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
        loss = self._loss(
            model_outputs[-1].reshape(
                -1,
            ),
            labels_t,
        )
        self._loss.reduction = prev_reduction

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()

    def compute_loss_from_predictions(  # type: ignore
        self, pred: np.ndarray, y: np.ndarray, reduction: str = "none", **kwargs
    ) -> np.ndarray | "torch.Tensor":
        """
        Compute the loss of the regressor for predictions `pred`. Does not apply preprocessing to the given `y`.

        :param pred: Model predictions.
        :param y: Target values.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                   'none': no reduction will be applied
                   'mean': the sum of the output will be divided by the number of elements in the output,
                   'sum': the output will be summed.
        :return: Loss values.
        """
        import torch

        if isinstance(y, torch.Tensor):
            labels_t = y
        else:
            # Convert the labels to Tensors
            labels_t = torch.from_numpy(y).to(self._device)

        prev_reduction = self._loss.reduction

        # Return individual loss values
        self._loss.reduction = reduction
        loss = self._loss(
            pred[-1].reshape(
                -1,
            ),
            labels_t,
        )
        self._loss.reduction = prev_reduction

        if isinstance(y, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()

    def compute_losses(
        self,
        x: np.ndarray | "torch.Tensor",
        y: np.ndarray | "torch.Tensor",
        reduction: str = "none",
    ) -> dict[str, np.ndarray | "torch.Tensor"]:
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
        x: np.ndarray | "torch.Tensor",
        y: np.ndarray | "torch.Tensor",
        training_mode: bool = False,
        **kwargs,
    ) -> np.ndarray | "torch.Tensor":
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
                              Note on RNN-like models: Backpropagation through RNN modules in eval mode raises
                              RuntimeError due to cudnn issues and require training mode, i.e. RuntimeError: cudnn RNN
                              backward can only be called in training mode. Therefore, if the model is an RNN type we
                              always use training mode but freeze batch-norm and dropout layers if
                              `training_mode=False.`
        :return: Array of gradients of the same shape as `x`.
        """
        import torch

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
        # y_preprocessed = self.reduce_labels(y_preprocessed)

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)
        else:
            labels_t = y_preprocessed

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        loss = self._loss(
            model_outputs[-1].reshape(
                -1,
            ),
            labels_t,
        )

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        if self._use_amp:  # pragma: no cover
            from apex import amp

            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            loss.backward()

        grads: torch.Tensor | np.ndarray

        if x_grad.grad is not None:
            if isinstance(x, torch.Tensor):
                grads = x_grad.grad
            else:
                grads = x_grad.grad.cpu().numpy().copy()
        else:
            raise ValueError("Gradient tensor in PyTorch model is `None`.")

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

    def custom_loss_gradient(
        self,
        loss_fn,
        x: np.ndarray | "torch.Tensor",
        y: np.ndarray | "torch.Tensor",
        layer_name,
        training_mode: bool = False,
    ) -> np.ndarray | "torch.Tensor":
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param loss_fn: Loss function w.r.t to which gradient needs to be calculated.
        :param x: Sample input with shape as expected by the model(base image).
        :param y: Sample input with shape as expected by the model(target image).
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.`
        :param layer_name: Name of the layer from which activation needs to be extracted/activation layer.
        :return: Array of gradients of the same shape as `x`.
        """
        import torch

        self._model.train(mode=training_mode)
        self._model.eval()
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
            inputs_t, _ = self._apply_preprocessing(x_grad, y=None, fit=False, no_grad=False)
            targets_t, _ = self._apply_preprocessing(y_grad, y=None, fit=False, no_grad=False)
        if isinstance(x, np.ndarray):
            x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False, no_grad=True)
            y_preprocessed, _ = self._apply_preprocessing(y, y=None, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            y_grad = torch.from_numpy(y_preprocessed).to(self._device)
            x_grad.requires_grad = True
            y_grad.requires_grad = False
            inputs_t = x_grad
            targets_t = y_grad
        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        # Compute the gradient and return
        model_outputs1 = self.get_activations(inputs_t, layer_name, 1, framework=True)
        model_outputs2 = self.get_activations(targets_t, layer_name, 1, framework=True)
        diff = model_outputs1 - model_outputs2
        loss = loss_fn(diff, p=2)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        if self._use_amp:  # pragma: no cover
            from apex import amp

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

    def get_activations(  # type: ignore
        self,
        x: np.ndarray | "torch.Tensor",
        layer: int | str | None = None,
        batch_size: int = 128,
        framework: bool = False,
    ) -> np.ndarray | "torch.Tensor":
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
        import torch

        self._model.eval()

        # Apply defences
        if framework:
            no_grad = False
        else:
            no_grad = True
        x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False, no_grad=no_grad)

        # Get index of the extracted layer
        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:  # pragma: no cover
                raise ValueError(f"Layer name {layer} not supported")
            layer_index = self._layer_names.index(layer)

        elif isinstance(layer, int):
            layer_index = layer

        else:  # pragma: no cover
            raise TypeError("Layer must be of type str or int")

        def get_feature(name):
            # the hook signature
            def hook(model, input, output):  # pylint: disable=redefined-builtin,unused-argument
                self._features[name] = output

            return hook

        if not hasattr(self, "_features"):
            self._features: dict[str, torch.Tensor] = {}
            # register forward hooks on the layers of choice

        if layer not in self._features:
            interim_layer = dict([*self._model._model.named_modules()])[self._layer_names[layer_index]]
            interim_layer.register_forward_hook(get_feature(self._layer_names[layer_index]))

        if framework:
            if isinstance(x_preprocessed, torch.Tensor):
                self._model(x_preprocessed)
                return self._features[self._layer_names[layer_index]]
            input_tensor = torch.from_numpy(x_preprocessed)
            self._model(input_tensor.to(self._device))
            return self._features[self._layer_names[layer_index]]

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
            self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            layer_output = self._features[self._layer_names[layer_index]]
            results.append(layer_output.detach().cpu().numpy())

        results_array = np.concatenate(results)

        return results_array

    def save(self, filename: str, path: str | None = None) -> None:
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        import torch

        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        # disable pylint because access to _modules required
        torch.save(self._model._model.state_dict(), full_path + ".model")
        torch.save(self._optimizer.state_dict(), full_path + ".optimizer")  # type: ignore
        logger.info("Model state dict saved in path: %s.", full_path + ".model")
        logger.info("Optimizer state dict saved in path: %s.", full_path + ".optimizer")

    def __getstate__(self) -> dict[str, Any]:
        """
        Use to ensure `PyTorchRegressor` can be pickled.

        :return: State dictionary with instance parameters.
        """

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

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Use to ensure `PyTorchRegressor` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        """
        import torch

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
            f"{self.__module__ + '.' + self.__class__.__name__}(model={self._model}, loss={self._loss}, "
            f"optimizer={self._optimizer}, input_shape={self._input_shape}, "
            f"channels_first={self.channels_first}, clip_values={self.clip_values!r}, "
            f"preprocessing_defences={self.preprocessing_defences}, "
            f"postprocessing_defences={self.postprocessing_defences}, preprocessing={self.preprocessing})"
        )

        return repr_

    def _make_model_wrapper(self, model: "torch.nn.Module") -> "torch.nn.Module":
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        try:
            import torch

            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(torch.nn.Module):
                    """
                    This is a wrapper for the input model.
                    """

                    import torch

                    def __init__(self, model: torch.nn.Module):
                        """
                        Initialization by storing the input model.

                        :param model: PyTorch model. The forward function of the model must return the logit output.
                        """
                        super().__init__()
                        self._model = model

                    # disable pylint because of API requirements for function
                    def forward(self, x):
                        """
                        This is where we get outputs from the input model.

                        :param x: Input data.
                        :type x: `torch.Tensor`
                        :return: a list of output layers, where the last 2 layers are logit and final outputs.
                        :rtype: `list`
                        """

                        # disable pylint because access to _model required

                        result = []
                        if isinstance(self._model, torch.nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
                                result.append(x)

                        elif isinstance(self._model, torch.nn.Module):
                            x = self._model(x)
                            result.append(x)

                        else:  # pragma: no cover
                            raise TypeError("The input model must inherit from `nn.Module`.")

                        return result

                    @property
                    def get_layers(self) -> list[str]:
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
                        import torch

                        result = []
                        if isinstance(self._model, torch.nn.Module):
                            for name, _ in self._model._modules.items():
                                result.append(name)

                        else:  # pragma: no cover
                            raise TypeError("The input model must inherit from `nn.Module`.")
                        logger.info(
                            "Inferred %i hidden layers on PyTorch regressor.",
                            len(result),
                        )

                        return result

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._model_wrapper(model)

        except ImportError:  # pragma: no cover
            raise ImportError("Could not find PyTorch (`torch`) installation.") from ImportError
