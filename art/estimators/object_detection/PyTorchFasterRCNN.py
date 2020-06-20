# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the task specific estimator for Faster R-CNN v3 in PyTorch.
"""
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.pytorch import PyTorchEstimator
from art.utils import Deprecated, deprecated_keyword_arg

if TYPE_CHECKING:
    import torchvision

    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchFasterRCNN(ObjectDetectorMixin, PyTorchEstimator):
    """
    This class implements a model-specific object detector using Faster-RCNN and PyTorch.
    """

    @deprecated_keyword_arg("channel_index", end_version="1.5.0", replaced_by="channels_first")
    def __init__(
        self,
        model: Optional["torchvision.models.detection.fasterrcnn_resnet50_fpn"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channel_index=Deprecated,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0, 1),
        attack_losses: Tuple[str, ...] = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg",),
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: Faster-RCNN model. The output of the model is `List[Dict[Tensor]]`, one for each input image. The
                      fields of the Dict are as follows:

                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                        between 0 and H and 0 and W
                      - labels (Int64Tensor[N]): the predicted labels for each image
                      - scores (Tensor[N]): the scores or each prediction
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch

        # Remove in 1.5.0
        if channel_index == 3:
            channels_first = False
        elif channel_index == 1:
            channels_first = True
        elif channel_index is not Deprecated:
            raise ValueError("Not a proper channel_index. Use channels_first.")

        super().__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")

        if self.preprocessing is not None:
            raise ValueError("This estimator does not support `preprocessing`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        if model is None:
            import torchvision  # lgtm [py/repeated-import]

            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
            )
        else:
            self._model = model

        # Set device
        self._device: str
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        self._model.to(self._device)
        self._model.eval()
        self.attack_losses: Tuple[str, ...] = attack_losses

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
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
        import torch
        import torchvision  # lgtm [py/repeated-import]

        self._model.train()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        if y is not None:
            for i, y_i in enumerate(y):
                y[i]["boxes"] = torch.tensor(y_i["boxes"], dtype=torch.float).to(self._device)
                y[i]["labels"] = torch.tensor(y_i["labels"], dtype=torch.int64).to(self._device)
                y[i]["scores"] = torch.tensor(y_i["scores"]).to(self._device)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list = list()

        for i in range(x.shape[0]):
            if self.clip_values is not None:
                img = transform(x[i] / self.clip_values[1]).to(self._device)
            else:
                img = transform(x[i]).to(self._device)
            img.requires_grad = True
            image_tensor_list.append(img)

        output = self._model(image_tensor_list, y)

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

        grad_list = list()
        for img in image_tensor_list:
            gradients = img.grad.cpu().numpy().copy()
            grad_list.append(gradients)

        grads = np.stack(grad_list, axis=0)

        # BB
        grads = self._apply_preprocessing_gradient(x, grads)
        grads = np.swapaxes(grads, 1, 3)
        grads = np.swapaxes(grads, 1, 2)
        assert grads.shape == x.shape

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        return grads

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[Tensor]]`, one for each input image. The
                 fields of the Dict are as follows:

                 - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                   between 0 and H and 0 and W
                 - labels (Int64Tensor[N]): the predicted labels for each image
                 - scores (Tensor[N]): the scores or each prediction.
        """
        import torchvision  # lgtm [py/repeated-import]

        self._model.eval()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list: List[np.ndarray] = list()

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0
        for i in range(x.shape[0]):
            image_tensor_list.append(transform(x[i] / norm_factor).to(self._device))
        predictions = self._model(image_tensor_list)
        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def set_learning_phase(self, train: bool) -> None:
        raise NotImplementedError
