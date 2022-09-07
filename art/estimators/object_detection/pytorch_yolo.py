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
This module implements the task specific estimator for PyTorch YOLO v3 object detectors.

| Paper link: https://arxiv.org/abs/1804.02767
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


def translate_predictions_xcycwh_to_x1y1x2y2(
    y_pred_xcycwh: "torch.Tensor", input_height: int, input_width: int
) -> List[Dict[str, "torch.Tensor"]]:
    """
    Convert object detection predictions from xcycwh to x1y1x2y2 format.

    :param y_pred_xcycwh: Labels in format xcycwh.
    :return: Labels in format x1y1x2y2.
    """
    import torch  # lgtm [py/repeated-import]

    y_pred_x1y1x2y2 = []

    for i in range(y_pred_xcycwh.shape[0]):

        y_i = {
            "boxes": torch.permute(
                torch.vstack(
                    [
                        torch.maximum(
                            (y_pred_xcycwh[i, :, 0] - y_pred_xcycwh[i, :, 2] / 2),
                            torch.tensor(0).to(y_pred_xcycwh.device),
                        ),
                        torch.maximum(
                            (y_pred_xcycwh[i, :, 1] - y_pred_xcycwh[i, :, 3] / 2),
                            torch.tensor(0).to(y_pred_xcycwh.device),
                        ),
                        torch.minimum(
                            (y_pred_xcycwh[i, :, 0] + y_pred_xcycwh[i, :, 2] / 2),
                            torch.tensor(input_height).to(y_pred_xcycwh.device),
                        ),
                        torch.minimum(
                            (y_pred_xcycwh[i, :, 1] + y_pred_xcycwh[i, :, 3] / 2),
                            torch.tensor(input_width).to(y_pred_xcycwh.device),
                        ),
                    ]
                ),
                (1, 0),
            ),
            "labels": torch.argmax(y_pred_xcycwh[i, :, 5:], dim=1, keepdim=False),
            "scores": y_pred_xcycwh[i, :, 4],
        }

        y_pred_x1y1x2y2.append(y_i)

    return y_pred_x1y1x2y2


def translate_labels_art_to_yolov3(labels_art: List[Dict[str, "torch.Tensor"]]):
    """
    Translate labels from ART to YOLO v3.

    :param labels_art: Object detection labels in format ART (torchvision).
    :return: Object detection labels in format YOLO v3.
    """
    import torch  # lgtm [py/repeated-import]

    yolo_targets_list = []

    for i_dict, label_dict in enumerate(labels_art):
        num_detectors = label_dict["boxes"].size()[0]
        targets = torch.zeros(num_detectors, 6)
        targets[:, 0] = i_dict
        targets[:, 1] = label_dict["labels"]
        targets[:, 2:6] = label_dict["boxes"]
        targets[:, 4] = targets[:, 4] - targets[:, 2]
        targets[:, 5] = targets[:, 5] - targets[:, 3]
        yolo_targets_list.append(targets)

    yolo_targets = torch.vstack(yolo_targets_list)

    return yolo_targets


class PyTorchYolo(ObjectDetectorMixin, PyTorchEstimator):
    """
    This module implements the model- and task specific estimator for YOLO v3 object detector models in PyTorch.

    | Paper link: https://arxiv.org/abs/1804.02767
    """

    estimator_params = PyTorchEstimator.estimator_params + ["attack_losses"]

    def __init__(
        self,
        model,
        input_shape=(3, 416, 416),
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = True,
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

                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                        between 0 and H and 0 and W
                      - labels (Int64Tensor[N]): the predicted labels for each image
                      - scores (Tensor[N]): the scores or each prediction
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: [Currently unused] Set channels first or last.
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
        import torch  # lgtm [py/repeated-import]
        import torchvision  # lgtm [py/repeated-import]

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
        )

        self._input_shape = input_shape

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This estimator requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:  # pragma: no cover
                raise ValueError("This estimator requires un-normalized input images with clip_vales=(0, max_value).")

        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        # Set device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f"cuda:{cuda_idx}")

        self._model.to(self._device)
        self._model.eval()
        self.attack_losses: Tuple[str, ...] = attack_losses

    @property
    def native_label_is_pytorch_format(self) -> bool:
        """
        Are the native labels in PyTorch format [x1, y1, x2, y2]?
        """
        return True

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    def _get_losses(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Tuple[Dict[str, "torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
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
        import torch  # lgtm [py/repeated-import]

        self._model.train()

        # Apply preprocessing
        if self.all_framework_preprocessing:

            if y is not None and isinstance(y, list) and isinstance(y[0]["boxes"], np.ndarray):
                y_tensor = []
                for y_i in y:
                    y_t = {
                        "boxes": torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device),
                        "labels": torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device),
                    }
                    if "masks" in y_i:
                        y_t["masks"] = torch.from_numpy(y_i["masks"]).type(torch.int64).to(self.device)
                    y_tensor.append(y_t)
            elif y is not None and isinstance(y, dict):
                y_tensor = []
                for i in range(y["boxes"].shape[0]):
                    y_t = {"boxes": y["boxes"][i], "labels": y["labels"][i]}
                    y_tensor.append(y_t)
            else:
                y_tensor = y  # type: ignore

            if isinstance(x, np.ndarray):
                if self.clip_values is not None:
                    norm_factor = self.clip_values[1]
                else:
                    norm_factor = 1.0

                x_grad = torch.from_numpy(x / norm_factor).to(self.device)
                x_grad.requires_grad = True

            else:
                x_grad = x.to(self.device)
                if x_grad.shape[2] < x_grad.shape[0] and x_grad.shape[2] < x_grad.shape[1]:
                    x_grad = torch.permute(x_grad, (2, 0, 1))

            image_tensor_list_grad = x_grad
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x_grad, y=y_tensor, fit=False, no_grad=False)
            inputs_t = x_preprocessed

        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)

            if y_preprocessed is not None and isinstance(y_preprocessed[0]["boxes"], np.ndarray):
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

            if self.clip_values is not None:
                norm_factor = self.clip_values[1]
            else:
                norm_factor = 1.0

            x_grad = torch.from_numpy(x_preprocessed / norm_factor).to(self.device)
            x_grad.requires_grad = True
            image_tensor_list_grad = x_grad
            inputs_t = image_tensor_list_grad

        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        labels_t = translate_labels_art_to_yolov3(labels_art=y_preprocessed)

        loss_components = self._model(inputs_t, labels_t)

        return loss_components, inputs_t, image_tensor_list_grad

    def loss_gradient(  # pylint: disable=W0613
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss gradients of the same shape as `x`.
        """
        output, inputs_t, image_tensor_list_grad = self._get_losses(x=x, y=y)

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
            grads = image_tensor_list_grad.grad.cpu().numpy().copy()
        else:
            grads = inputs_t.grad.copy()

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict
                 are as follows:

                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image
                 - scores [N]: the scores or each prediction.
        """
        import torch  # lgtm [py/repeated-import]

        x = x.copy()

        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0

        x_preprocessed_tensor = torch.from_numpy(x_preprocessed).to(self.device)
        x_preprocessed_tensor /= norm_factor

        predictions_xcycwh = self._model(x_preprocessed_tensor)

        predictions_x1y1x2y2_tensor = translate_predictions_xcycwh_to_x1y1x2y2(
            y_pred_xcycwh=predictions_xcycwh, input_height=self.input_shape[1], input_width=self.input_shape[2]
        )
        predictions_x1y1x2y2_array: List[Dict[str, np.ndarray]] = []

        for i_prediction, _ in enumerate(predictions_x1y1x2y2_tensor):
            predictions_x1y1x2y2_array.append({})

            predictions_x1y1x2y2_array[i_prediction]["boxes"] = (
                predictions_x1y1x2y2_tensor[i_prediction]["boxes"].detach().cpu().numpy()
            )
            predictions_x1y1x2y2_array[i_prediction]["labels"] = (
                predictions_x1y1x2y2_tensor[i_prediction]["labels"].detach().cpu().numpy()
            )
            predictions_x1y1x2y2_array[i_prediction]["scores"] = (
                predictions_x1y1x2y2_tensor[i_prediction]["scores"].detach().cpu().numpy()
            )
            if "masks" in predictions_x1y1x2y2_tensor[i_prediction]:
                predictions_x1y1x2y2_array[i_prediction]["masks"] = (
                    predictions_x1y1x2y2_tensor[i_prediction]["masks"].detach().cpu().numpy().squeeze()
                )

        return predictions_x1y1x2y2_array

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        raise NotImplementedError

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

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
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
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss.
        """
        import torch  # lgtm [py/repeated-import]

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
