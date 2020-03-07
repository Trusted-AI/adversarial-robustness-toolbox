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

from art.estimators.pytorch import PyTorchEstimator
from art.estimators.object_detectors.object_detector import ObjectDetectorMixin

logger = logging.getLogger(__name__)


class PyTorchFasterRCNN(ObjectDetectorMixin, PyTorchEstimator):
    def __init__(self, model=None, clip_values=None, channel_index=None, preprocessing_defences=None,
                 postprocessing_defences=None, preprocessing=None):

        super().__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        if model is None:
            import torchvision

            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
            )
        else:
            self._model = model
        self._model.eval()

    def fit(self):
        raise NotImplementedError

    def get_activations(self):
        raise NotImplementedError

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """

        import torch
        import numpy as np

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(np.argmax(y_preprocessed, axis=1)).to(self._device)

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        loss = self._loss(model_outputs[-1], labels_t)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = inputs_t.grad.cpu().numpy().copy()
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads


    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        import torchvision
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image = transform(x)
        image_list = [image]
        predictions = self._model(image_list)

        return predictions

    def set_learning_phase(self):
        raise NotImplementedError
