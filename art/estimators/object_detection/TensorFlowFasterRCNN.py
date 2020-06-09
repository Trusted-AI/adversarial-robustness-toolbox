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
This module implements the task specific estimator for Faster R-CNN in TensorFlow.
"""
import logging

import numpy as np
from object_detection.utils import config_util
from object_detection.builders import model_builder

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.tensorflow import TensorFlowEstimator
from art.utils import Deprecated, deprecated_keyword_arg
from art.utils import get_file
from art.config import ART_DATA_PATH

logger = logging.getLogger(__name__)


class TensorFlowFasterRCNN(ObjectDetectorMixin, TensorFlowEstimator):
    """
    This class implements a model-specific object detector using Faster-RCNN and TensorFlow.
    """

    @deprecated_keyword_arg("channel_index", end_version="1.5.0", replaced_by="channels_first")
    def __init__(
        self,
        model=None,
        filename='faster_rcnn_inception_v2_coco_2017_11_08',
        url='http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2017_11_08.tar.gz',
        clip_values=None,
        channel_index=Deprecated,
        channels_first=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
    ):
        """
        Initialization.

        :param model: Faster-RCNN model. The output of the model is `List[Dict[Tensor]]`, one for each input image. The
                      fields of the Dict are as follows:

                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
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
        :param channels_first: Set channels first or last.
        :type channels_first: `bool`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :type attack_losses: `Tuple[str]`
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        :type device_type: `string`
        """
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

        assert clip_values[0] == 0, "This classifier requires un-normalized input images with clip_vales=(0, max_value)"
        assert clip_values[1] > 0, "This classifier requires un-normalized input images with clip_vales=(0, max_value)"
        assert preprocessing is None, "This estimator does not support `preprocessing`."
        assert postprocessing_defences is None, "This estimator does not support `postprocessing_defences`."

        if model is None:
            # Download and extract
            path = get_file(filename=filename, path=ART_DATA_PATH, url=url, extract=True)

            # Load model config
            pipeline_config = path + '/pipeline.config'
            configs = config_util.get_configs_from_pipeline_file(pipeline_config)
            configs['model'].faster_rcnn.second_stage_batch_size = configs[
                'model'].faster_rcnn.first_stage_max_proposals

            # Load model
            self._model = model_builder.build(model_config=configs['model'], is_training=True, add_summaries=False)

        else:
            self._model = model

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :type x: `np.ndarray`
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
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

        if y is not None:
            for i, y_i in enumerate(y):
                y[i]["boxes"] = torch.FloatTensor(y_i["boxes"]).to(self._device)
                y[i]["labels"] = torch.LongTensor(y_i["labels"]).to(self._device)
                y[i]["scores"] = torch.Tensor(y_i["scores"]).to(self._device)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        image_tensor_list = list()

        for i in range(x.shape[0]):
            img = transform(x[i] / self.clip_values[1]).to(self._device)
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

        grads = grads / self.clip_values[1]

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

                 - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                   between 0 and H and 0 and W
                 - labels (Int64Tensor[N]): the predicted labels for each image
                 - scores (Tensor[N]): the scores or each prediction.
        :rtype: `np.ndarray`
        """
        import torchvision

        self._model.eval()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        image_tensor_list = list()

        for i in range(x.shape[0]):
            image_tensor_list.append(transform(x[i] / self.clip_values[1]).to(self._device))
        predictions = self._model(image_tensor_list)
        return predictions

    def fit(self):
        raise NotImplementedError

    def get_activations(self):
        raise NotImplementedError

    def set_learning_phase(self):
        raise NotImplementedError
