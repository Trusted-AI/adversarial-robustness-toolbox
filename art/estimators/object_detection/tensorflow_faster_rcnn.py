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
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import variables_helper

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.tensorflow import TensorFlowEstimator
from art.utils import Deprecated, deprecated_keyword_arg
from art.utils import get_file
from art.config import ART_DATA_PATH

if TYPE_CHECKING:
    from object_detection.meta_architectures.faster_rcnn_meta_arch import FasterRCNNMetaArch
    from tensorflow.python.framework.ops import Tensor
    from tensorflow.python.client.session import Session

    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowFasterRCNN(ObjectDetectorMixin, TensorFlowEstimator):
    """
    This class implements a model-specific object detector using Faster-RCNN and TensorFlow.
    """

    @deprecated_keyword_arg("channel_index", end_version="1.5.0", replaced_by="channels_first")
    def __init__(
        self,
        model: Optional["FasterRCNNMetaArch"] = None,
        filename: Optional[str] = None,
        url: Optional[str] = None,
        images: Optional["Tensor"] = None,
        sess: Optional["Session"] = None,
        is_training: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channel_index=Deprecated,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0, 1),
        attack_losses: Tuple[str, ...] = (
            'first_stage_localization_loss',
            'first_stage_objectness_loss',
            'second_stage_localization_loss',
            'second_stage_classification_loss'
        )
    ):
        """
        Initialization of an instance TensorFlowFasterRCNN.

        :param model: A TensorFlow Faster-RCNN model. The output that can be computed from the model includes a tuple
               of (predictions, losses, detections):

                    - predictions: a dictionary holding "raw" prediction tensors.
                    - losses: a dictionary mapping loss keys (`first_stage_localization_loss`,
                    `first_stage_objectness_loss`, `second_stage_localization_loss`,
                    `second_stage_classification_loss`) to scalar tensors representing
                    corresponding loss values.
                    - detections: a dictionary containing final detection results.
        :param filename: Name of the file.
        :param url: Download URL.
        :param images: Input samples of shape (nb_samples, height, width, nb_channels).
        :param sess: Computation session.
        :param is_training: A boolean indicating whether the training version of the computation graph should be
               constructed.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channel_index: Index of the axis in data containing the color channels or features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of the following loss components:
               `first_stage_localization_loss`, `first_stage_objectness_loss`, `second_stage_localization_loss`,
               `second_stage_classification_loss`.
        """
        # Remove in 1.5.0
        if channel_index == 3:
            channels_first = False
        elif channel_index == 1:
            channels_first = True
        elif channel_index is not Deprecated:
            raise ValueError("Not a proper channel_index. Use channels_first.")

        # Super initialization
        super().__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        # Check clip values
        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This classifier requires normalized input images with clip_vales=(0, 1).")
            if self.clip_values[1] != 1:
                raise ValueError("This classifier requires normalized input images with clip_vales=(0, 1).")

        # Check preprocessing and postprocessing_defences
        if self.preprocessing is not None:
            raise ValueError("This estimator does not support `preprocessing`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        # Create placeholders for groundtruth boxes
        self._groundtruth_boxes_list: List["Tensor"]
        self._groundtruth_boxes_list = [
            tf.placeholder(
                dtype=tf.float32,
                shape=(None, 4),
                name="groundtruth_boxes_{}".format(i)
            ) for i in range(images.shape[0])
        ]

        # Create placeholders for groundtruth classes
        self._groundtruth_classes_list: List["Tensor"]
        self._groundtruth_classes_list = [
            tf.placeholder(
                dtype=tf.int32,
                shape=(None,),
                name="groundtruth_classes_{}".format(i)
            ) for i in range(images.shape[0])
        ]

        # Create placeholders for groundtruth weights
        self._groundtruth_weights_list: List["Tensor"]
        self._groundtruth_weights_list = [
            tf.placeholder(
                dtype=tf.float32,
                shape=(None,),
                name="groundtruth_weights_{}".format(i)
            ) for i in range(images.shape[0])
        ]

        # Load model
        if model is None:
            # If model is None, then we need to have parameters filename and url to download, extract and load the
            # object detection model
            if filename is None or url is None:
                filename, url = (
                    'faster_rcnn_inception_v2_coco_2017_11_08',
                    'http://download.tensorflow.org/models/object_detection/'
                    'faster_rcnn_inception_v2_coco_2017_11_08.tar.gz'
                )

            self._predictions, self._losses, self._detections = self._load_model(
                filename=filename,
                url=url,
                obj_detection_model=None,
                images=images,
                is_training=is_training,
                groundtruth_boxes_list=self._groundtruth_boxes_list,
                groundtruth_classes_list=self._groundtruth_classes_list,
                groundtruth_weights_list=self._groundtruth_weights_list
            )

        else:
            self._predictions, self._losses, self._detections = self._load_model(
                filename=None,
                url=None,
                obj_detection_model=model,
                images=images,
                is_training=is_training,
                groundtruth_boxes_list=self._groundtruth_boxes_list,
                groundtruth_classes_list=self._groundtruth_classes_list,
                groundtruth_weights_list=self._groundtruth_weights_list
            )

        # Save new attributes
        self._is_training: bool = is_training
        self._images: Optional["Tensor"] = images
        self._attack_losses: Tuple[str, ...] = attack_losses

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
        self._sess: Optional["Session"] = sess

    @staticmethod
    def _load_model(
        filename: Optional[str] = None,
        url: Optional[str] = None,
        obj_detection_model: Optional["FasterRCNNMetaArch"] = None,
        images: Optional["Tensor"] = None,
        is_training: bool = False,
        groundtruth_boxes_list: Optional[List["Tensor"]] = None,
        groundtruth_classes_list: Optional[List["Tensor"]] = None,
        groundtruth_weights_list: Optional[List["Tensor"]] = None
    ) -> Tuple[Dict["Tensor"], Dict["Tensor"], Dict["Tensor"]]:
        """
        Download, extract and load a model from a URL if it not already in the cache. The file at indicated by `url`
        is downloaded to the path ~/.art/data and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip
        formats will also be extracted. Then the model is loaded, pipelined and its outputs are returned as a tuple
        of (predictions, losses, detections).

        :param filename: Name of the file.
        :param url: Download URL.
        :param images: Input samples of shape (nb_samples, height, width, nb_channels).
        :param is_training: A boolean indicating whether the training version of the computation graph should be
               constructed.
        :param groundtruth_boxes_list: a list of 2-D tf.float32 tensors of shape [num_boxes, 4] containing
               coordinates of the groundtruth boxes. Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
               format and assumed to be normalized and clipped relative to the image window with y_min <= y_max and
               x_min <= x_max.
        :param groundtruth_classes_list: a list of 1-D tf.float32 tensors of shape [num_boxes] containing the class
               targets with the zero index assumed to map to the first non-background class.
        :param groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape [num_boxes] containing weights for
               groundtruth boxes.
        :return: A tuple of (predictions, losses, detections):

                    - predictions: a dictionary holding "raw" prediction tensors.
                    - losses: a dictionary mapping loss keys (`first_stage_localization_loss`,
                    `first_stage_objectness_loss`, `second_stage_localization_loss`,
                    `second_stage_classification_loss`) to scalar tensors representing
                    corresponding loss values.
                    - detections: a dictionary containing final detection results.
        """
        if obj_detection_model is None:
            # If obj_detection_model is None, then we need to have parameters filename and url to download, extract
            # and load the object detection model
            if filename is None or url is None:
                raise ValueError(
                    "Need input parameters `filename` and `url` to download, "
                    "extract and load the object detection model."
                )

            # Download and extract
            path = get_file(filename=filename, path=ART_DATA_PATH, url=url, extract=True)

            # Load model config
            pipeline_config = path + '/pipeline.config'
            configs = config_util.get_configs_from_pipeline_file(pipeline_config)
            configs['model'].faster_rcnn.second_stage_batch_size = configs[
                'model'
            ].faster_rcnn.first_stage_max_proposals

            # Load model
            obj_detection_model = model_builder.build(
                model_config=configs['model'],
                is_training=is_training,
                add_summaries=False
            )

        # Provide groundtruth
        groundtruth_classes_list = [
            tf.one_hot(
                groundtrue_class,
                obj_detection_model.num_classes
            ) for groundtrue_class in groundtruth_classes_list
        ]

        obj_detection_model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list,
            groundtruth_weights_list=groundtruth_weights_list
        )

        # Create model pipeline
        images *= 255.0
        preprocessed_images, true_image_shapes = obj_detection_model.preprocess(images)
        predictions = obj_detection_model.predict(preprocessed_images, true_image_shapes)
        losses = obj_detection_model.loss(predictions, true_image_shapes)
        detections = obj_detection_model.postprocess(predictions, true_image_shapes)

        # Initialize variables from checkpoint
        # Get variables to restore
        variables_to_restore = obj_detection_model.restore_map(
            fine_tune_checkpoint_type='detection',
            load_all_detection_checkpoint_vars=True
        )

        # Get variables from checkpoint
        fine_tune_checkpoint_path = path + '/model.ckpt'
        vars_in_ckpt = variables_helper.get_variables_available_in_checkpoint(
            variables_to_restore,
            fine_tune_checkpoint_path,
            include_global_step=False
        )

        # Initialize from checkpoint
        tf.train.init_from_checkpoint(fine_tune_checkpoint_path, vars_in_ckpt)

        return predictions, losses, detections

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :type x: `np.ndarray`
        :param y: A dictionary of target values. The fields of the dictionary are as follows:

                    - `groundtruth_boxes_list`: a list of `nb_samples` size of 2-D tf.float32 tensors of shape
                    [num_boxes, 4] containing coordinates of the groundtruth boxes. Groundtruth boxes are provided in
                    [y_min, x_min, y_max, x_max] format and assumed to be normalized and clipped relative to the image
                    window with y_min <= y_max and x_min <= x_max.
                    - `groundtruth_classes_list`: a list of `nb_samples` size of 1-D tf.float32 tensors of shape
                    [num_boxes] containing the class targets with the zero index assumed to map to the first
                    non-background class.
                    - `groundtruth_weights_list`: A list of `nb_samples` size of 1-D tf.float32 tensors of shape
                    [num_boxes] containing weights for groundtruth boxes.
        :type y: `dict`
        :return: Loss gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # Only do loss_gradient if is_training is False
        if self._is_training:
            raise NotImplementedError(
                "This object detector was loaded in training mode and therefore not support loss_gradient."
            )

        # Only do loss_gradient if model is internally loaded
        if not (
            hasattr(self, '_groundtruth_boxes_list') and
            hasattr(self, '_groundtruth_classes_list') and
            hasattr(self, '_groundtruth_weights_list')
        ):
            raise NotImplementedError(
                "This object detector was loaded externally and therefore not support loss_gradient."
            )

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Get the loss gradients graph
        if not hasattr(self, '_loss_grads'):
            loss = None
            for loss_name in self._attack_losses:
                if loss is None:
                    loss = self._losses[loss_name]
                else:
                    loss = loss + self._losses[loss_name]

            self._loss_grads = tf.gradients(loss, self._images)[0]

        # Create feed_dict
        feed_dict = {self._images: x_preprocessed}

        for (placeholder, value) in zip(self._groundtruth_boxes_list, y['groundtruth_boxes_list']):
            feed_dict[placeholder] = value

        for (placeholder, value) in zip(self._groundtruth_classes_list, y['groundtruth_classes_list']):
            feed_dict[placeholder] = value

        for (placeholder, value) in zip(self._groundtruth_weights_list, y['groundtruth_weights_list']):
            feed_dict[placeholder] = value

        # Compute gradients
        grads = self._sess.run(self._loss_grads, feed_dict=feed_dict)
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :type x: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :return: A dictionary containing the following fields:

                    - detection_boxes: `[batch, max_detection, 4]`
                    - detection_scores: `[batch, max_detections]`
                    - detection_classes: `[batch, max_detections]`
                    - detection_multiclass_scores: `[batch, max_detections, 2]`
                    - detection_anchor_indices: `[batch, max_detections]`
                    - num_detections: `[batch]`
                    - raw_detection_boxes: `[batch, total_detections, 4]`
                    - raw_detection_scores: `[batch, total_detections, num_classes + 1]`
        :rtype: `dict`
        """
        # Only do prediction if is_training is False
        if self._is_training:
            raise NotImplementedError(
                "This object detector was loaded in training mode and therefore not support prediction."
            )

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        num_samples = x.shape[0]
        results = {
            'detection_boxes': np.zeros((
                num_samples,
                self._detections['detection_boxes'].shape[1].value,
                self._detections['detection_boxes'].shape[2].value
            ), dtype=np.float32),
            'detection_scores': np.zeros((
                num_samples,
                self._detections['detection_scores'].shape[1].value
            ), dtype=np.float32),
            'detection_classes': np.zeros((
                num_samples,
                self._detections['detection_classes'].shape[1].value
            ), dtype=np.float32),
            'detection_multiclass_scores': np.zeros((
                num_samples,
                self._detections['detection_multiclass_scores'].shape[1].value,
                self._detections['detection_multiclass_scores'].shape[2].value
            ), dtype=np.float32),
            'detection_anchor_indices': np.zeros((
                num_samples,
                self._detections['detection_anchor_indices'].shape[1].value
            ), dtype=np.float32),
            'num_detections': np.zeros((
                num_samples,
            ), dtype=np.float32),
            'raw_detection_boxes': np.zeros((
                num_samples,
                self._detections['raw_detection_boxes'].shape[1].value,
                self._detections['raw_detection_boxes'].shape[2].value
            ), dtype=np.float32),
            'raw_detection_scores': np.zeros((
                num_samples,
                self._detections['raw_detection_scores'].shape[1].value,
                self._detections['raw_detection_scores'].shape[2].value
            ), dtype=np.float32)
        }

        num_batch = int(np.ceil(num_samples / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, num_samples)

            # Create feed_dict
            feed_dict = {self._images: x[begin:end]}

            # Run prediction
            batch_results = self._sess.run(self._detections, feed_dict=feed_dict)

            # Update final results
            results['detection_boxes'][begin:end] = batch_results['detection_boxes']
            results['detection_scores'][begin:end] = batch_results['detection_scores']
            results['detection_classes'][begin:end] = batch_results['detection_classes']
            results['detection_multiclass_scores'][begin:end] = batch_results['detection_multiclass_scores']
            results['detection_anchor_indices'][begin:end] = batch_results['detection_anchor_indices']
            results['num_detections'][begin:end] = batch_results['num_detections']
            results['raw_detection_boxes'][begin:end] = batch_results['raw_detection_boxes']
            results['raw_detection_scores'][begin:end] = batch_results['raw_detection_scores']

        return results

    @property
    def predictions(self):
        """
        Get the `_predictions` attribute.

        :return: A dictionary holding "raw" prediction tensors.
        :rtype: `dict`
        """
        return self._predictions

    @property
    def losses(self):
        """
        Get the `_losses` attribute.

        :return: A dictionary mapping loss keys (`first_stage_localization_loss`, `first_stage_objectness_loss`,
        `second_stage_localization_loss`, `second_stage_classification_loss`) to scalar tensors representing
        corresponding loss values.
        :rtype: `dict`
        """
        return self._losses

    @property
    def detections(self):
        """
        Get the `_detections` attribute.

        :return: A dictionary containing final detection results.
        :rtype: `dict`
        """
        return self._detections

    def fit(self):
        raise NotImplementedError

    def get_activations(self):
        raise NotImplementedError

    def set_learning_phase(self):
        raise NotImplementedError
