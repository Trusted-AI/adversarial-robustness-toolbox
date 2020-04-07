# MIT License
#
# Copyright (C) IBM Corporation 2020
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

import numpy as np

from art.estimators.pytorch import PyTorchEstimator
from art.estimators.object_detection.object_detector import ObjectDetectorMixin

logger = logging.getLogger(__name__)


class PyTorchFasterRCNN(ObjectDetectorMixin, PyTorchEstimator):
    """
    This class implements a model-specific object detector using Faster-RCNN and PyTorch.
    """

    def __init__(
        self,
        model=None,
        clip_values=None,
        channel_index=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
        attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
    ):
        """
        Initialization.

        :param model: Faster-RCNN model. The output of the model is `List[Dict[Tensor]]`, one for each input image. The
                      fields of the Dict are as follows:
                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                        between 0 and H and 0 and W
                      - labels (Int64Tensor[N]): the predicted labels for each image
                      - scores (Tensor[N]): the scores or each prediction
        :type model: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param attack_losses: Tuple of any combinaiton of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :type attack_losses: `Tuple[str]`
        """

        super().__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        assert preprocessing is None, "This estimator does not support `preprocessing`."
        assert postprocessing_defences is None, "This estimator does not support `postprocessing_defences`."

        if model is None:
            import torchvision

            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
            )
        else:
            self._model = model

        # Set device
        import torch

        if torch.cuda.is_available():
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

        self._model.eval()

        self.attack_losses = attack_losses

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :type x: `np.ndarray`
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:
                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :type y: `np.ndarray`
        :return: Loss gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import torch
        import torchvision

        self._model.train()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)
        x = x.astype(np.uint8)

        if y is not None:
            for i, y_i in enumerate(y):
                y[i]["boxes"] = torch.FloatTensor(y_i["boxes"]).to(self._device)
                y[i]["labels"] = torch.LongTensor(y_i["labels"]).to(self._device)
                y[i]["scores"] = torch.Tensor(y_i["scores"]).to(self._device)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        image_tensor_list = list()

        for i in range(x.shape[0]):
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

        print("loss", loss)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward(retain_graph=True)

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

        return grads

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :type x: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :return: Predictions of format `List[Dict[Tensor]]`, one for each input image. The
                 fields of the Dict are as follows:
                 - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values
                   between 0 and H and 0 and W
                 - labels (Int64Tensor[N]): the predicted labels for each image
                 - scores (Tensor[N]): the scores or each prediction.
        :rtype: `np.ndarray`
        """
        import torchvision

        self._model.eval()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)
        x = x.astype(np.uint8)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        image_tensor_list = list()

        for i in range(x.shape[0]):
            image_tensor_list.append(transform(x[i]).to(self._device))
        predictions = self._model(image_tensor_list)
        return predictions

    def fit(self):
        raise NotImplementedError

    def get_activations(self):
        raise NotImplementedError

    def set_learning_phase(self):
        raise NotImplementedError
