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
        import torchvision
        import numpy as np

        self._model.train()

        # For training
        # images = torch.rand(4, 3, 600, 1200)
        # boxes = torch.rand(4, 11, 4)
        # labels = torch.randint(1, 91, (4, 11))

        # images = list(image for image in images)

        # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # image = transform(x)
        # images = [image]

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        image_tensor_list = list()

        for i in range(x.shape[0]):
            img = transform(x[i])
            # print('print(x[i].shape)')
            # print(x[i].shape)
            # print('img')
            # print(img.shape)
            img.requires_grad = True
            image_tensor_list.append(img)

        if y is None:
            predictions = self.predict(x=x)
        else:
            predictions = y

        self._model.train()

        # predictions[0]["labels"].numpy())]  # Get the Prediction Score
        # print(pred_class)
        # pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in
        #               list(predictions[0]["boxes"].detach().numpy())]  # Bounding boxes
        # pred_score = list(predictions[0]["scores"].

        targets = []

        for i in range(len(image_tensor_list)):
            d = {}
            # d['boxes'] = boxes[i]
            d['boxes'] = predictions[i]["boxes"]
            # d['labels'] = labels[i]
            d['labels'] = predictions[i]["labels"]
            targets.append(d)

        output = self._model(image_tensor_list, targets)

        print(output)
        print(output['loss_classifier'])

        # Apply preprocessing
        # x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Convert the inputs to Tensors
        # inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        # inputs_t.requires_grad = True

        # Convert the labels to Tensors
        # labels_t = torch.from_numpy(np.argmax(y_preprocessed, axis=1)).to(self._device)

        # Compute the gradient and return
        # model_outputs = self._model(inputs_t)
        # loss = self._loss(model_outputs[-1], labels_t)
        loss = output['loss_classifier']

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        # grads = inputs_t.grad.cpu().numpy().copy()
        # grads = image.grad.cpu().numpy().copy()
        grad_list = list()
        for img in image_tensor_list:
            gradients = img.grad.cpu().numpy().copy()
            grad_list.append(gradients)

        grads = np.stack(grad_list, axis=0)

        # grads = self._apply_preprocessing_gradient(x, grads)

        # print(grads.shape)
        # print(type(grads))

        # if len(image_tensor_list) == 1:
        #     grads = np.newaxis(grads, axis=0)

        # print(grads.shape)
        grads = np.swapaxes(grads, 1, 3)
        grads = np.swapaxes(grads, 1, 2)
        # print(x.shape)
        # print(grads.shape)
        assert grads.shape == x.shape

        return grads

    def predict(self, x, **kwargs):
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

        self._model.eval()

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        image_tensor_list = list()

        for i in range(x.shape[0]):
            image_tensor_list.append(transform(x[i]))
        predictions = self._model(image_tensor_list)
        return predictions

    def set_learning_phase(self):
        raise NotImplementedError
