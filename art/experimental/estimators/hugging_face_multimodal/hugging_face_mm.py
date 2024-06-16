# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements an estimator for attacking pre-trained CLIP by adversarial perturbations on the image.
| Paper link: https://arxiv.org/abs/2103.00020
"""
import logging
import random
from typing import List, Optional, Any, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from art.estimators.pytorch import PyTorchEstimator


if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    import transformers
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.experimental.estimators.hugging_face_multimodal import HuggingFaceMultiModalInput

logger = logging.getLogger(__name__)


class HuggingFaceMultiModalPyTorch(PyTorchEstimator):
    """
    This module implements an estimator for attacking pre-trained CLIP by adversarial perturbations on the image.
    Currently only supports PGD attacks.
    """

    estimator_params = PyTorchEstimator.estimator_params + ["input_shape", "optimizer"]

    def __init__(
        self,
        model: "transformers.PreTrainedModel",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        optimizer: Optional["torch.optim.Optimizer"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: CLIP model
        :param input_shape: The shape of one input sample.
        :param optimizer: The optimizer for training the classifier.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
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
        self._optimizer = optimizer
        self.loss_fn = loss
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")
        self._model = model
        self._model: torch.nn.Module
        self._model.to(self._device)
        self._model.eval()

        # Attributes for forward compatibility with progress bar updates.
        self.training_loss: List[Any] = []
        self.training_accuracy: List[Any] = []

    @property
    def model(self) -> "torch.nn.Module":
        """
        Return the model.

        :return: The model.
        """
        return self._model

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape

    @property
    def optimizer(self) -> Optional["torch.optim.Optimizer"]:
        """
        Return the optimizer.

        :return: The optimizer.
        """
        return self._optimizer

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    @staticmethod
    def _preprocess_and_convert_inputs(
        x: "HuggingFaceMultiModalInput",
        y: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        fit: bool = False,  # pylint: disable=W0613
        no_grad: bool = True,  # pylint: disable=W0613
    ) -> Tuple["HuggingFaceMultiModalInput", Union[np.ndarray, "torch.Tensor", None]]:
        """
        Dummy function to allow compatibility with ART attacks.
        All pre-processing should be done before by the relevant HF pre-processor.

        :param x: Dictionary inputs for the CLIP model.
        :param y: Labels
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :param no_grad: `True` if no gradients required.
        :return: Preprocessed inputs `(x, y)
        """
        return x, y

    def _get_losses(self, x: "HuggingFaceMultiModalInput", y: Union[np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
        """
        Get the loss tensor output of the model including all preprocessing.

        :param x: Dictionary inputs for the CLIP model.
        :param y: Labels for the loss
        :return: Loss components and gradients of the input `x`.
        """
        import torch

        self._model.eval()

        if isinstance(y, np.ndarray):
            y = torch.tensor(y)

        x = x.to(self._device)
        y = y.to(self._device)

        # reduce labels
        if y.ndim > 1:
            y = torch.argmax(y, dim=-1)
        # Set gradients again after inputs are moved to another device
        if x["pixel_values"].is_leaf:
            x["pixel_values"].requires_grad = True  # type: ignore
        else:
            x["pixel_values"].retain_grad()

        # Calculate loss components
        preds = self._model(**x)
        preds = preds.logits_per_image
        return self.loss_fn(preds, y)

    def loss_gradient(  # pylint: disable=W0613
        self, x: "HuggingFaceMultiModalInput", y: Union[np.ndarray, "torch.Tensor"], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. the image component of the input

        :param x: Dictionary inputs for the CLIP model.
        :param y: Labels for the loss
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported
                       and providing it takes no effect.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch

        loss = self._get_losses(x=x, y=y)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()  # type: ignore
        x_grad = x["pixel_values"].grad

        if x_grad is not None:
            assert isinstance(x_grad, torch.Tensor)
            grads = x_grad.clone()
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.channels_first:
            grads = torch.permute(grads, (0, 2, 3, 1))

        assert grads.shape == x["pixel_values"].shape
        return grads.cpu().numpy()

    def predict(
        self, x: Union["HuggingFaceMultiModalInput", np.ndarray], batch_size: int = 128, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Dictionary inputs for the CLIP model.
        :param batch_size: Batch size.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported
                       and providing it takes no effect.
        :return: Predictions over the supplied data.
        """
        from art.experimental.estimators.hugging_face_multimodal.hugging_face_mm_inputs import (
            HuggingFaceMultiModalInput,
        )

        # Set model to evaluation mode
        self._model.eval()
        if isinstance(x, np.ndarray):
            raise ValueError("x should be of type HuggingFaceMultiModalInput")
        x_preprocessed, _ = self._preprocess_and_convert_inputs(x=x, y=None, fit=False, no_grad=True)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        results = []
        for m in tqdm(range(num_batch)):
            x_batch = x[batch_size * m : batch_size * (m + 1)]
            x_batch = x_batch.to(self._device)
            if isinstance(x_batch, HuggingFaceMultiModalInput):
                predictions = self._model(**x_batch)
            else:
                raise ValueError("expected type HuggingFaceMultiModalInput")
            results.append(predictions.logits_per_image.cpu().detach().numpy())

        return np.concatenate(results)

    def fit(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "HuggingFaceMultiModalInput"],
        y: Union[np.ndarray, "torch.Tensor"],
        batch_size: int = 128,
        nb_epochs: int = 10,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        display_progress_bar: bool = True,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) in index labels style of shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param display_progress_bar: Displays the training progress.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported
                       and providing it takes no effect.
        """
        import torch
        from art.experimental.estimators.hugging_face_multimodal.hugging_face_mm_inputs import (
            HuggingFaceMultiModalInput,
        )

        self._model.train()
        if self._optimizer is None:
            raise ValueError("Please supply a optimizer")

        y_tensor = torch.from_numpy(y)

        num_batch = int(len(y_tensor) / float(batch_size))
        ind = np.arange(len(y_tensor)).tolist()

        # Start training
        for _ in tqdm(range(nb_epochs)):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            pbar = tqdm(range(num_batch), disable=not display_progress_bar)
            accs = []
            losses = []

            for m in pbar:
                x_batch = x[ind[batch_size * m : batch_size * (m + 1)]]
                y_batch = y_tensor[ind[batch_size * m : batch_size * (m + 1)]]

                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                try:
                    if isinstance(x_batch, HuggingFaceMultiModalInput):
                        model_outputs = self._model(**x_batch)
                    else:
                        raise ValueError("expected type HuggingFaceMultiModalInput")
                except ValueError as err:
                    if "Expected more than 1 value per channel when training" in str(err):
                        logger.exception(
                            "Try dropping the last incomplete batch by setting drop_last=True in "
                            "method PyTorchClassifier.fit."
                        )
                    raise err

                loss = self.loss_fn(model_outputs["logits_per_image"], y_batch)

                loss.backward()

                self._optimizer.step()
                losses.append(loss.data.detach().cpu().numpy())

                if isinstance(y_batch, torch.Tensor):
                    y_batch = y_batch.detach().cpu().numpy()

                acc = np.sum(
                    np.argmax(model_outputs["logits_per_image"].detach().cpu().numpy(), axis=1) == y_batch
                ) / len(y_batch)
                accs.append(acc)

                if display_progress_bar:
                    pbar.set_description(f"Loss {np.mean(np.stack(losses)):.2f} " f"Acc {np.mean(np.stack(accs)):.2f} ")

            self.training_loss.append(losses)
            self.training_accuracy.append(accs)

            if scheduler is not None:
                scheduler.step()

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_loss(  # type: ignore
        self, x: "HuggingFaceMultiModalInput", y: Union[np.ndarray, "torch.Tensor"], **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Dictionary inputs for the CLIP model.
        :param y: Target values
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported
                       and providing it takes no effect.
        :return: Loss.
        """
        import torch

        loss, _ = self._get_losses(x=x, y=y)

        assert loss is not None

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()
