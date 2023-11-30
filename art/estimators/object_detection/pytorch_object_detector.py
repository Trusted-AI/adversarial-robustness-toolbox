# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements the task specific estimator for PyTorch object detectors.
"""
import logging
from typing import Any, List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.object_detection.utils import cast_inputs_to_pt
from art.estimators.pytorch import PyTorchEstimator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchObjectDetector(ObjectDetectorMixin, PyTorchEstimator):
    """
    This module implements the task specific estimator for PyTorch object detection models following the input and
    output formats of torchvision.
    """

    estimator_params = PyTorchEstimator.estimator_params + ["input_shape", "optimizer", "attack_losses"]

    def __init__(
        self,
        model: "torch.nn.Module",
        input_shape: Tuple[int, ...] = (-1, -1, -1),
        optimizer: Optional["torch.optim.Optimizer"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: bool = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: Object detection model. The output of the model is `List[Dict[str, torch.Tensor]]`, one for
                      each input image. The fields of the Dict are as follows:

                      - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                        0 <= y1 < y2 <= H.
                      - labels [N]: the labels for each image.
                      - scores [N]: the scores of each prediction.
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
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch
        import torchvision

        torch_version = list(map(int, torch.__version__.lower().split("+", maxsplit=1)[0].split(".")))
        torchvision_version = list(map(int, torchvision.__version__.lower().split("+", maxsplit=1)[0].split(".")))
        assert not (torch_version[0] == 1 and (torch_version[1] == 8 or torch_version[1] == 9)), (
            "PyTorchObjectDetector does not support torch==1.8 and torch==1.9 because of "
            "https://github.com/pytorch/vision/issues/4153. Support will return for torch==1.10."
        )
        assert not (torchvision_version[0] == 0 and (torchvision_version[1] == 9 or torchvision_version[1] == 10)), (
            "PyTorchObjectDetector does not support torchvision==0.9 and torchvision==0.10 because of "
            "https://github.com/pytorch/vision/issues/4153. Support will return for torchvision==0.11."
        )

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
        self._attack_losses = attack_losses

        # Parameters used for subclasses
        self.weight_dict: Optional[Dict[str, float]] = None
        self.criterion: Optional[torch.nn.Module] = None

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:  # pragma: no cover
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")

        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        self._model: torch.nn.Module
        self._model.to(self._device)
        self._model.eval()

    @property
    def native_label_is_pytorch_format(self) -> bool:
        """
        Are the native labels in PyTorch format [x1, y1, x2, y2]?
        """
        return True

    @property
    def model(self) -> "torch.nn.Module":
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
    def attack_losses(self) -> Tuple[str, ...]:
        """
        Return the combination of strings of the loss components.

        :return: The combination of strings of the loss components.
        """
        return self._attack_losses

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    def _preprocess_and_convert_inputs(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Optional[List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]] = None,
        fit: bool = False,
        no_grad: bool = True,
    ) -> Tuple["torch.Tensor", List[Dict[str, "torch.Tensor"]]]:
        """
        Apply preprocessing on inputs `(x, y)` and convert to tensors, if needed.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :param no_grad: `True` if no gradients required.
        :return: Preprocessed inputs `(x, y)` as tensors.
        """
        import torch

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0

        if self.all_framework_preprocessing:
            # Convert samples into tensor
            x_tensor, y_tensor = cast_inputs_to_pt(x, y)

            if not self.channels_first:
                x_tensor = torch.permute(x_tensor, (0, 3, 1, 2))
            x_tensor = x_tensor / norm_factor

            # Set gradients
            if not no_grad:
                if x_tensor.is_leaf:
                    x_tensor.requires_grad = True
                else:
                    x_tensor.retain_grad()

            # Apply framework-specific preprocessing
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x=x_tensor, y=y_tensor, fit=fit, no_grad=no_grad)

        elif isinstance(x, np.ndarray):
            # Apply preprocessing
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x=x, y=y, fit=fit, no_grad=no_grad)

            # Convert inputs into tensor
            x_preprocessed, y_preprocessed = cast_inputs_to_pt(x_preprocessed, y_preprocessed)

            if not self.channels_first:
                x_preprocessed = torch.permute(x_preprocessed, (0, 3, 1, 2))
            x_preprocessed = x_preprocessed / norm_factor

            # Set gradients
            if not no_grad:
                x_preprocessed.requires_grad = True

        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        return x_preprocessed, y_preprocessed

    def _translate_labels(self, labels: List[Dict[str, "torch.Tensor"]]) -> Any:
        """
        Translate object detection labels from ART format (torchvision) to the model format (torchvision) and
        move tensors to GPU, if applicable.

        :param labels: Object detection labels in format x1y1x2y2 (torchvision).
        :return: Object detection labels in format x1y1x2y2 (torchvision).
        """
        labels_translated = [{k: v.to(self.device) for k, v in y_i.items()} for y_i in labels]
        return labels_translated

    def _translate_predictions(self, predictions: Any) -> List[Dict[str, np.ndarray]]:  # pylint: disable=R0201
        """
        Translate object detection predictions from the model format (torchvision) to ART format (torchvision) and
        convert tensors to numpy arrays.

        :param predictions: Object detection predictions in format x1y1x2y2 (torchvision).
        :return: Object detection predictions in format x1y1x2y2 (torchvision).
        """
        predictions_x1y1x2y2: List[Dict[str, np.ndarray]] = []
        for pred in predictions:
            prediction = {}

            prediction["boxes"] = pred["boxes"].detach().cpu().numpy()
            prediction["labels"] = pred["labels"].detach().cpu().numpy()
            prediction["scores"] = pred["scores"].detach().cpu().numpy()
            if "masks" in pred:
                prediction["masks"] = pred["masks"].detach().cpu().numpy().squeeze()

            predictions_x1y1x2y2.append(prediction)

        return predictions_x1y1x2y2

    def _get_losses(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Tuple[Dict[str, "torch.Tensor"], "torch.Tensor"]:
        """
        Get the loss tensor output of the model including all preprocessing.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss components and gradients of the input `x`.
        """
        self._model.train()

        self.set_dropout(False)
        self.set_multihead_attention(False)

        # Apply preprocessing and convert to tensors
        x_preprocessed, y_preprocessed = self._preprocess_and_convert_inputs(x=x, y=y, fit=False, no_grad=False)

        # Move inputs to device
        x_preprocessed = x_preprocessed.to(self.device)
        y_preprocessed = self._translate_labels(y_preprocessed)

        # Set gradients again after inputs are moved to another device
        if x_preprocessed.is_leaf:
            x_preprocessed.requires_grad = True
        else:
            x_preprocessed.retain_grad()

        if self.criterion is None:
            loss_components = self._model(x_preprocessed, y_preprocessed)
        else:
            outputs = self._model(x_preprocessed)
            loss_components = self.criterion(outputs, y_preprocessed)

        return loss_components, x_preprocessed

    def loss_gradient(  # pylint: disable=W0613
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch

        loss_components, x_grad = self._get_losses(x=x, y=y)

        # Compute the loss
        if self.weight_dict is None:
            loss = sum(loss_components[loss_name] for loss_name in self.attack_losses if loss_name in loss_components)
        else:
            loss = sum(
                loss_component * self.weight_dict[loss_name]
                for loss_name, loss_component in loss_components.items()
                if loss_name in self.weight_dict
            )

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward(retain_graph=True)  # type: ignore

        if x_grad.grad is not None:
            if isinstance(x, np.ndarray):
                grads = x_grad.grad.cpu().numpy()
            else:
                grads = x_grad.grad.clone()
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        if not self.channels_first:
            if isinstance(x, np.ndarray):
                grads = np.transpose(grads, (0, 2, 3, 1))
            else:
                grads = torch.permute(grads, (0, 2, 3, 1))

        assert grads.shape == x.shape

        return grads

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape NCHW or NHWC.
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict
                 are as follows:

                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image.
                 - scores [N]: the scores or each prediction.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Set model to evaluation mode
        self._model.eval()

        # Apply preprocessing and convert to tensors
        x_preprocessed, _ = self._preprocess_and_convert_inputs(x=x, y=None, fit=False, no_grad=True)

        # Create dataloader
        dataset = TensorDataset(x_preprocessed)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        predictions: List[Dict[str, np.ndarray]] = []
        for (x_batch,) in dataloader:
            # Move inputs to device
            x_batch = x_batch.to(self._device)

            # Run prediction
            with torch.no_grad():
                outputs = self._model(x_batch)

            predictions_x1y1x2y2 = self._translate_predictions(outputs)
            predictions.extend(predictions_x1y1x2y2)

        return predictions

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]],
        batch_size: int = 128,
        nb_epochs: int = 10,
        drop_last: bool = False,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
        """
        import torch
        from torch.utils.data import Dataset, DataLoader

        # Set model to train mode
        self._model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        # Apply preprocessing and convert to tensors
        x_preprocessed, y_preprocessed = self._preprocess_and_convert_inputs(x=x, y=y, fit=True, no_grad=True)

        class ObjectDetectionDataset(Dataset):
            """
            Object detection dataset in PyTorch.
            """

            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        # Create dataloader
        dataset = ObjectDetectionDataset(x_preprocessed, y_preprocessed)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            collate_fn=lambda batch: list(zip(*batch)),
        )

        # Start training
        for _ in range(nb_epochs):
            # Train for one epoch
            for x_batch, y_batch in dataloader:
                # Move inputs to device
                x_batch = torch.stack(x_batch).to(self.device)
                y_batch = self._translate_labels(y_batch)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Get the loss components
                if self.criterion is None:
                    loss_components = self._model(x_batch, y_batch)
                else:
                    outputs = self._model(x_batch)
                    loss_components = self.criterion(outputs, y_batch)

                # Form the loss tensor
                if self.weight_dict is None:
                    loss = sum(
                        loss_components[loss_name] for loss_name in self.attack_losses if loss_name in loss_components
                    )
                else:
                    loss = sum(
                        loss_component * self.weight_dict[loss_name]
                        for loss_name, loss_component in loss_components.items()
                        if loss_name in self.weight_dict
                    )

                # Do training
                loss.backward()  # type: ignore
                self._optimizer.step()

            if scheduler is not None:
                scheduler.step()

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_losses(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all loss components.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Dictionary of loss components.
        """
        loss_components, _ = self._get_losses(x=x, y=y)
        output = {}
        for key, value in loss_components.items():
            if key in self.attack_losses:
                output[key] = value.detach().cpu().numpy()
        return output

    def compute_loss(  # type: ignore
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss.
        """
        import torch

        loss_components, _ = self._get_losses(x=x, y=y)

        # Compute the loss
        if self.weight_dict is None:
            loss = sum(loss_components[loss_name] for loss_name in self.attack_losses if loss_name in loss_components)
        else:
            loss = sum(
                loss_component * self.weight_dict[loss_name]
                for loss_name, loss_component in loss_components.items()
                if loss_name in self.weight_dict
            )

        assert isinstance(loss, torch.Tensor)

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()
