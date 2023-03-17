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
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
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
        channels_first: Optional[bool] = False,
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

        :param model: Object detection model. The output of the model is `List[Dict[Tensor]]`, one for each input
                      image. The fields of the Dict are as follows:

                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                        between 0 and H and 0 and W
                      - labels (Int64Tensor[N]): the predicted labels for each image
                      - scores (Tensor[N]): the scores or each prediction
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

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:  # pragma: no cover
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")

        if preprocessing is not None:
            raise ValueError("This estimator does not support `preprocessing`.")
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

    def _get_losses(
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Tuple[Dict[str, "torch.Tensor"], List["torch.Tensor"], List["torch.Tensor"]]:
        """
        Get the loss tensor output of the model including all preprocessing.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The fields of the Dict are as
                  follows:

                  - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                               0 <= y1 < y2 <= H.
                  - labels (Int64Tensor[N]): the labels for each image
        :return: Loss gradients of the same shape as `x`.
        """
        import torch
        import torchvision

        self._model.train()

        # Apply preprocessing
        if self.all_framework_preprocessing:

            if y is not None and isinstance(y, list) and isinstance(y[0]["boxes"], np.ndarray):
                y_tensor = []
                for i, y_i in enumerate(y):
                    y_t = {}
                    y_t["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device)
                    y_t["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device)
                    if "masks" in y_i:
                        y_t["masks"] = torch.from_numpy(y_i["masks"]).type(torch.int64).to(self.device)
                    y_tensor.append(y_t)
            elif y is not None and isinstance(y, dict):
                y_tensor = []
                for i in range(y["boxes"].shape[0]):
                    y_t = {}
                    y_t["boxes"] = y["boxes"][i]
                    y_t["labels"] = y["labels"][i]
                    y_tensor.append(y_t)
            else:
                y_tensor = y  # type: ignore

            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image_tensor_list_grad = []
            y_preprocessed = []
            inputs_t = []

            for i in range(x.shape[0]):
                if isinstance(x, np.ndarray):
                    if self.clip_values is not None:
                        x_grad = transform(x[i] / self.clip_values[1]).to(self.device)
                    else:
                        x_grad = transform(x[i]).to(self.device)
                    x_grad.requires_grad = True
                else:
                    x_grad = x[i].to(self.device)
                    if x_grad.shape[2] < x_grad.shape[0] and x_grad.shape[2] < x_grad.shape[1]:
                        x_grad = torch.permute(x_grad, (2, 0, 1))

                image_tensor_list_grad.append(x_grad)
                x_grad_1 = torch.unsqueeze(x_grad, dim=0)
                x_preprocessed_i, y_preprocessed_i = self._apply_preprocessing(
                    x_grad_1, y=[y_tensor[i]], fit=False, no_grad=False
                )
                for i_preprocessed in range(x_preprocessed_i.shape[0]):
                    inputs_t.append(x_preprocessed_i[i_preprocessed])
                    y_preprocessed.append(y_preprocessed_i[i_preprocessed])

        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)

            if y_preprocessed is not None and isinstance(y_preprocessed[0]["boxes"], np.ndarray):
                y_preprocessed_tensor = []
                for i, y_i in enumerate(y_preprocessed):
                    y_preprocessed_t = {}
                    y_preprocessed_t["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device)
                    y_preprocessed_t["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device)
                    if "masks" in y_i:
                        y_preprocessed_t["masks"] = torch.from_numpy(y_i["masks"]).type(torch.uint8).to(self.device)
                    y_preprocessed_tensor.append(y_preprocessed_t)
                y_preprocessed = y_preprocessed_tensor

            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image_tensor_list_grad = []

            for i in range(x_preprocessed.shape[0]):
                if self.clip_values is not None:
                    x_grad = transform(x_preprocessed[i] / self.clip_values[1]).to(self.device)
                else:
                    x_grad = transform(x_preprocessed[i]).to(self.device)
                x_grad.requires_grad = True
                image_tensor_list_grad.append(x_grad)

            inputs_t = image_tensor_list_grad

        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self.device)  # type: ignore
        else:
            labels_t = y_preprocessed  # type: ignore

        output = self._model(inputs_t, labels_t)

        return output, inputs_t, image_tensor_list_grad

    def loss_gradient(  # pylint: disable=W0613
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch

        grad_list = []

        # Adding this loop because torch==[1.7, 1.8] and related versions of torchvision do not allow loss gradients at
        #  the input for batches larger than 1 anymore for PyTorch FasterRCNN because of a view created by torch or
        #  torchvision. This loop should be revisited with later releases of torch and removed once it becomes
        #  unnecessary.
        for i in range(x.shape[0]):

            x_i = x[[i]]
            y_i = [y[i]]

            output, inputs_t, image_tensor_list_grad = self._get_losses(x=x_i, y=y_i)

            # Compute the gradient and return
            loss = None
            for loss_name in self.attack_losses:
                if loss is None:
                    loss = output[loss_name]
                else:
                    loss = loss + output[loss_name]

            # Clean gradients
            self._model.zero_grad()

            # Compute gradients
            loss.backward(retain_graph=True)  # type: ignore

            if isinstance(x, np.ndarray):
                for img in image_tensor_list_grad:
                    if img.grad is not None:
                        gradients = img.grad.cpu().numpy().copy()
                    else:
                        raise ValueError("Gradient term in PyTorch model is `None`.")
                    grad_list.append(gradients)
            else:
                for img in inputs_t:
                    if img.grad is not None:
                        gradients = img.grad.copy()
                    else:
                        raise ValueError("Gradient term in PyTorch model is `None`.")
                    grad_list.append(gradients)

        if isinstance(x, np.ndarray):
            grads = np.stack(grad_list, axis=0)
            grads = np.transpose(grads, (0, 2, 3, 1))
        else:
            grads = torch.stack(grad_list, dim=0)
            grads = grads.premute(0, 2, 3, 1)

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

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
                 - labels [N]: the labels for each image
                 - scores [N]: the scores or each prediction.
        """
        import torch
        import torchvision

        # Set model to evaluation mode
        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Convert samples into tensors
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0

        if self.channels_first:
            x_preprocessed = [torch.from_numpy(x_i / norm_factor).to(self.device) for x_i in x_preprocessed]
        else:
            x_preprocessed = [transform(x_i / norm_factor).to(self.device) for x_i in x_preprocessed]

        predictions: List[Dict[str, np.ndarray]] = []

        # Run prediction
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch using indices
            i_batch = x_preprocessed[m * batch_size : (m + 1) * batch_size]

            with torch.no_grad():
                predictions_x1y1x2y2 = self._model(i_batch)

            for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                prediction = {}

                prediction["boxes"] = prediction_x1y1x2y2["boxes"].detach().cpu().numpy()
                prediction["labels"] = prediction_x1y1x2y2["labels"].detach().cpu().numpy()
                prediction["scores"] = prediction_x1y1x2y2["scores"].detach().cpu().numpy()
                if "masks" in prediction_x1y1x2y2:
                    prediction["masks"] = prediction_x1y1x2y2["masks"].detach().cpu().numpy().squeeze()

                predictions.append(prediction)

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
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
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
        import torchvision

        # Set model to train mode
        self._model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)

        # Convert samples into tensors
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0

        if self.channels_first:
            x_preprocessed = torch.from_numpy(x_preprocessed / norm_factor).to(self.device)
        else:
            x_preprocessed = torch.stack([transform(x_i / norm_factor) for x_i in x_preprocessed]).to(self.device)

        # Convert labels into tensors, if needed
        if isinstance(y_preprocessed[0]["boxes"], np.ndarray):
            y_preprocessed_tensor = []
            for y_i in y_preprocessed:
                y_preprocessed_t = {
                    "boxes": torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device),
                    "labels": torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device),
                }
                if "masks" in y_i:
                    y_preprocessed_t["masks"] = torch.from_numpy(y_i["masks"]).type(torch.uint8).to(self.device)
                y_preprocessed_tensor.append(y_preprocessed_t)
            y_preprocessed = y_preprocessed_tensor
        y_preprocessed = np.asarray(y_preprocessed)

        num_batch = len(x_preprocessed) / float(batch_size)
        if drop_last:
            num_batch = int(np.floor(num_batch))
        else:
            num_batch = int(np.ceil(num_batch))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            np.random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]
                o_batch = y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Form the loss function
                loss_components = self._model(i_batch, o_batch)
                if isinstance(loss_components, dict):
                    loss = sum(loss_components.values())
                else:
                    loss = loss_components

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
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all loss components.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Dictionary of loss components.
        """
        output_tensor, _, _ = self._get_losses(x=x, y=y)
        output = {}
        for key, value in output_tensor.items():
            output[key] = value.detach().cpu().numpy()
        return output

    def compute_loss(  # type: ignore
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss.
        """
        import torch

        output, _, _ = self._get_losses(x=x, y=y)

        # Compute the gradient and return
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = output[loss_name]
            else:
                loss = loss + output[loss_name]

        assert loss is not None

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()
