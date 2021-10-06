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

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.tensorflow import TensorFlowEstimator
from art.utils import get_file
from art import config

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow.compat.v1 as tf
    from object_detection.meta_architectures.faster_rcnn_meta_arch import FasterRCNNMetaArch
    from tensorflow.python.client.session import Session  # pylint: disable=E0611

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowFasterRCNN(ObjectDetectorMixin, TensorFlowEstimator):
    """
    This class implements a model-specific object detector using Faster-RCNN and TensorFlow.
    """

    estimator_params = TensorFlowEstimator.estimator_params + ["images", "sess", "is_training", "attack_losses"]

    def __init__(
        self,
        images: "tf.Tensor",
        model: Optional["FasterRCNNMetaArch"] = None,
        filename: Optional[str] = None,
        url: Optional[str] = None,
        sess: Optional["Session"] = None,
        is_training: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: bool = False,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        attack_losses: Tuple[str, ...] = (
            "Loss/RPNLoss/localization_loss",
            "Loss/RPNLoss/objectness_loss",
            "Loss/BoxClassifierLoss/localization_loss",
            "Loss/BoxClassifierLoss/classification_loss",
        ),
    ):
        """
        Initialization of an instance TensorFlowFasterRCNN.

        :param images: Input samples of shape (nb_samples, height, width, nb_channels).
        :param model: A TensorFlow Faster-RCNN model. The output that can be computed from the model includes a tuple
                      of (predictions, losses, detections):

                        - predictions: a dictionary holding "raw" prediction tensors.
                        - losses: a dictionary mapping loss keys (`Loss/RPNLoss/localization_loss`,
                                  `Loss/RPNLoss/objectness_loss`, `Loss/BoxClassifierLoss/localization_loss`,
                                  `Loss/BoxClassifierLoss/classification_loss`) to scalar tensors representing
                                  corresponding loss values.
                        - detections: a dictionary containing final detection results.
        :param filename: Filename of the detection model without filename extension.
        :param url: URL to download archive of detection model including filename extension.
        :param sess: Computation session.
        :param is_training: A boolean indicating whether the training version of the computation graph should be
                            constructed.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
                            maximum values allowed for input image features. If floats are provided, these will be
                            used as the range of all features. If arrays are provided, each value will be considered
                            the bound for a feature, thus the shape of clip values needs to match the total number
                            of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
                              used for data preprocessing. The first value will be subtracted from the input. The
                              input will then be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of the following loss components:
                              `first_stage_localization_loss`, `first_stage_objectness_loss`,
                              `second_stage_localization_loss`, `second_stage_classification_loss`.
        """
        import tensorflow.compat.v1 as tf  # lgtm [py/repeated-import]

        # Super initialization
        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == 0):
                raise ValueError("This classifier requires normalized input images with clip_vales=(0, 1).")
            if not np.all(self.clip_values[1] == 1):  # pragma: no cover
                raise ValueError("This classifier requires normalized input images with clip_vales=(0, 1).")

        # Check preprocessing and postprocessing defences
        if self.preprocessing_defences is not None:
            raise ValueError("This estimator does not support `preprocessing_defences`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        # Create placeholders for groundtruth boxes
        self._groundtruth_boxes_list: List["tf.Tensor"]
        self._groundtruth_boxes_list = [
            tf.placeholder(dtype=tf.float32, shape=(None, 4), name="groundtruth_boxes_{}".format(i))
            for i in range(images.shape[0])
        ]

        # Create placeholders for groundtruth classes
        self._groundtruth_classes_list: List["tf.Tensor"]
        self._groundtruth_classes_list = [
            tf.placeholder(dtype=tf.int32, shape=(None,), name="groundtruth_classes_{}".format(i))
            for i in range(images.shape[0])
        ]

        # Create placeholders for groundtruth weights
        self._groundtruth_weights_list: List["tf.Tensor"]
        self._groundtruth_weights_list = [
            tf.placeholder(dtype=tf.float32, shape=(None,), name="groundtruth_weights_{}".format(i))
            for i in range(images.shape[0])
        ]

        # Load model
        if model is None:
            # If model is None, then we need to have parameters filename and url to download, extract and load the
            # object detection model
            if filename is None or url is None:
                filename, url = (
                    "faster_rcnn_inception_v2_coco_2017_11_08",
                    "http://download.tensorflow.org/models/object_detection/"
                    "faster_rcnn_inception_v2_coco_2017_11_08.tar.gz",
                )

            self._model, self._predictions, self._losses, self._detections = self._load_model(
                images=images,
                filename=filename,
                url=url,
                obj_detection_model=None,
                is_training=is_training,
                groundtruth_boxes_list=self._groundtruth_boxes_list,
                groundtruth_classes_list=self._groundtruth_classes_list,
                groundtruth_weights_list=self._groundtruth_weights_list,
            )

        else:
            self._model, self._predictions, self._losses, self._detections = self._load_model(
                images=images,
                filename=None,
                url=None,
                obj_detection_model=model,
                is_training=is_training,
                groundtruth_boxes_list=self._groundtruth_boxes_list,
                groundtruth_classes_list=self._groundtruth_classes_list,
                groundtruth_weights_list=self._groundtruth_weights_list,
            )

        # Save new attributes
        self._input_shape = images.shape.as_list()[1:]
        self.is_training: bool = is_training
        self.images: Optional["tf.Tensor"] = images
        self.attack_losses: Tuple[str, ...] = attack_losses

        # Assign session
        if sess is None:
            logger.warning("A session cannot be None, create a new session.")
            self._sess = tf.Session()
        else:  # pragma: no cover
            self._sess = sess

        # Initialize variables
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())

    @property
    def native_label_is_pytorch_format(self) -> bool:
        """
        Are the native labels in PyTorch format [x1, y1, x2, y2]?
        """
        return False

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @staticmethod
    def _load_model(
        images: "tf.Tensor",
        filename: Optional[str] = None,
        url: Optional[str] = None,
        obj_detection_model: Optional["FasterRCNNMetaArch"] = None,
        is_training: bool = False,
        groundtruth_boxes_list: Optional[List["tf.Tensor"]] = None,
        groundtruth_classes_list: Optional[List["tf.Tensor"]] = None,
        groundtruth_weights_list: Optional[List["tf.Tensor"]] = None,
    ) -> Tuple[Dict[str, "tf.Tensor"], ...]:
        """
        Download, extract and load a model from a URL if it not already in the cache. The file at indicated by `url`
        is downloaded to the path ~/.art/data and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip
        formats will also be extracted. Then the model is loaded, pipelined and its outputs are returned as a tuple
        of (predictions, losses, detections).

        :param images: Input samples of shape (nb_samples, height, width, nb_channels).
        :param filename: Name of the file.
        :param url: Download URL.
        :param is_training: A boolean indicating whether the training version of the computation graph should be
                            constructed.
        :param groundtruth_boxes_list: A list of 2-D tf.float32 tensors of shape [num_boxes, 4] containing
                                       coordinates of the groundtruth boxes. Groundtruth boxes are provided in
                                       [y_min, x_min, y_max, x_max] format and also assumed to be normalized and
                                       clipped relative to the image window with conditions y_min <= y_max and
                                       x_min <= x_max.
        :param groundtruth_classes_list: A list of 1-D tf.float32 tensors of shape [num_boxes] containing the class
                                         targets with the zero index which is assumed to map to the first
                                         non-background class.
        :param groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape [num_boxes] containing weights for
                                         groundtruth boxes.
        :return: A tuple of (predictions, losses, detections):

                    - predictions: a dictionary holding "raw" prediction tensors.
                    - losses: a dictionary mapping loss keys (`Loss/RPNLoss/localization_loss`,
                              `Loss/RPNLoss/objectness_loss`, `Loss/BoxClassifierLoss/localization_loss`,
                              `Loss/BoxClassifierLoss/classification_loss`) to scalar tensors representing
                              corresponding loss values.
                    - detections: a dictionary containing final detection results.
        """
        import tensorflow.compat.v1 as tf  # lgtm [py/repeated-import]
        from object_detection.utils import variables_helper

        if obj_detection_model is None:
            from object_detection.utils import config_util
            from object_detection.builders import model_builder

            # If obj_detection_model is None, then we need to have parameters filename and url to download, extract
            # and load the object detection model
            if filename is None or url is None:  # pragma: no cover
                raise ValueError(
                    "Need input parameters `filename` and `url` to download, "
                    "extract and load the object detection model."
                )

            # Download and extract
            path = get_file(filename=filename, path=config.ART_DATA_PATH, url=url, extract=True)

            # Load model config
            pipeline_config = path + "/pipeline.config"
            configs = config_util.get_configs_from_pipeline_file(pipeline_config)
            configs["model"].faster_rcnn.second_stage_batch_size = configs[
                "model"
            ].faster_rcnn.first_stage_max_proposals

            # Load model
            obj_detection_model = model_builder.build(
                model_config=configs["model"], is_training=is_training, add_summaries=False
            )

        # Provide groundtruth
        if groundtruth_classes_list is not None:
            groundtruth_classes_list = [
                tf.one_hot(groundtruth_class, obj_detection_model.num_classes)
                for groundtruth_class in groundtruth_classes_list
            ]

        obj_detection_model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list,
            groundtruth_weights_list=groundtruth_weights_list,
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
            fine_tune_checkpoint_type="detection", load_all_detection_checkpoint_vars=True
        )

        # Get variables from checkpoint
        fine_tune_checkpoint_path = path + "/model.ckpt"
        vars_in_ckpt = variables_helper.get_variables_available_in_checkpoint(
            variables_to_restore, fine_tune_checkpoint_path, include_global_step=False
        )

        # Initialize from checkpoint
        tf.train.init_from_checkpoint(fine_tune_checkpoint_path, vars_in_ckpt)

        return obj_detection_model, predictions, losses, detections

    def loss_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, y: List[Dict[str, np.ndarray]], standardise_output: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Targets of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict are
                  as follows:

                 - boxes [N, 4]: the boxes in [y1, x1, y2, x2] in scale [0, 1] (`standardise_output=False`) or
                                 [x1, y1, x2, y2] in image scale (`standardise_output=True`) format,
                                 with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image in TensorFlow (`standardise_output=False`) or PyTorch
                               (`standardise_output=True`) format

        :param standardise_output: True if `y` is provided in standardised PyTorch format. Box coordinates will be
                                   scaled back to [0, 1], label index will be decreased by 1 and the boxes will be
                                   changed from [x1, y1, x2, y2] to [y1, x1, y2, x2] format, with
                                   0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        :return: Loss gradients of the same shape as `x`.
        """
        import tensorflow.compat.v1 as tf  # lgtm [py/repeated-import]

        # Only do loss_gradient if is_training is False
        if self.is_training:
            raise NotImplementedError(
                "This object detector was loaded in training mode and therefore not support loss_gradient."
            )

        if standardise_output:
            from art.estimators.object_detection.utils import convert_pt_to_tf

            y = convert_pt_to_tf(y=y, height=x.shape[1], width=x.shape[2])

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Get the loss gradients graph
        if not hasattr(self, "_loss_grads"):
            loss = None
            for loss_name in self.attack_losses:
                if loss is None:
                    loss = self._losses[loss_name]
                else:
                    loss = loss + self._losses[loss_name]

            self._loss_grads: tf.Tensor = tf.gradients(loss, self.images)[0]

        # Create feed_dict
        feed_dict = {self.images: x_preprocessed}

        for (placeholder, value) in zip(self._groundtruth_boxes_list, y):
            feed_dict[placeholder] = value["boxes"]

        for (placeholder, value) in zip(self._groundtruth_classes_list, y):
            feed_dict[placeholder] = value["labels"]

        for (placeholder, value) in zip(self._groundtruth_weights_list, y):
            feed_dict[placeholder] = [1.0] * len(value["labels"])

        # Compute gradients
        grads = self._sess.run(self._loss_grads, feed_dict=feed_dict)
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads

    def predict(  # pylint: disable=W0221
        self, x: np.ndarray, batch_size: int = 128, standardise_output: bool = False, **kwargs
    ) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param batch_size: Batch size.
        :param standardise_output: True if output should be standardised to PyTorch format. Box coordinates will be
                                   scaled from [0, 1] to image dimensions, label index will be increased by 1 to adhere
                                   to COCO categories and the boxes will be changed to [x1, y1, x2, y2] format, with
                                   0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The
                 fields of the Dict are as follows:

                 - boxes [N, 4]: the boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                                 Can be changed to PyTorch format with `standardise_output=True`.
                 - labels [N]: the labels for each image in TensorFlow format. Can be changed to PyTorch format with
                               `standardise_output=True`.
                 - scores [N]: the scores or each prediction.
        """
        # Only do prediction if is_training is False
        if self.is_training:
            raise NotImplementedError(
                "This object detector was loaded in training mode and therefore not support prediction."
            )

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Check if batch processing is appropriately set
        if self.images is not None and self.images.shape[0].value is not None:
            if x.shape[0] % self.images.shape[0].value != 0:  # pragma: no cover
                raise ValueError("Number of prediction samples must be a multiple of input size.")

            logger.warning("Reset batch size to input size.")
            batch_size = self.images.shape[0].value

        # Run prediction with batch processing
        num_samples = x.shape[0]
        num_batch = int(np.ceil(num_samples / float(batch_size)))
        results = list()

        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, num_samples)

            # Create feed_dict
            feed_dict = {self.images: x[begin:end]}

            # Run prediction
            batch_results = self._sess.run(self._detections, feed_dict=feed_dict)

            for i in range(end - begin):
                d_sample = dict()

                d_sample["boxes"] = batch_results["detection_boxes"][i]
                d_sample["labels"] = batch_results["detection_classes"][i].astype(np.int32)

                if standardise_output:
                    from art.estimators.object_detection.utils import convert_tf_to_pt

                    d_sample = convert_tf_to_pt(y=[d_sample], height=x.shape[1], width=x.shape[2])[0]

                d_sample["scores"] = batch_results["detection_scores"][i]

                results.append(d_sample)

        return results

    @property
    def input_images(self) -> "tf.Tensor":
        """
        Get the `images` attribute.

        :return: The input image tensor.
        """
        return self.images

    @property
    def predictions(self) -> Dict[str, "tf.Tensor"]:
        """
        Get the `_predictions` attribute.

        :return: A dictionary holding "raw" prediction tensors.
        """
        return self._predictions

    @property
    def losses(self) -> Dict[str, "tf.Tensor"]:
        """
        Get the `_losses` attribute.

        :return: A dictionary mapping loss keys (`Loss/RPNLoss/localization_loss`, `Loss/RPNLoss/objectness_loss`,
                 `Loss/BoxClassifierLoss/localization_loss`, `Loss/BoxClassifierLoss/classification_loss`) to scalar
                 tensors representing corresponding loss values.
        """
        return self._losses

    @property
    def detections(self) -> Dict[str, "tf.Tensor"]:
        """
        Get the `_detections` attribute.

        :return: A dictionary containing final detection results.
        """
        return self._detections

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Array of losses of the same shape as `x`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Get the losses graph
        if not hasattr(self, "_loss_total"):
            loss = None
            for loss_name in self.attack_losses:
                if loss is None:
                    loss = self._losses[loss_name]
                else:
                    loss = loss + self._losses[loss_name]
            self._loss_total = loss

        # Create feed_dict
        feed_dict = {self.images: x_preprocessed}

        for (placeholder, value) in zip(self._groundtruth_boxes_list, y):
            feed_dict[placeholder] = value["boxes"]

        for (placeholder, value) in zip(self._groundtruth_classes_list, y):
            feed_dict[placeholder] = value["labels"]

        for (placeholder, value) in zip(self._groundtruth_weights_list, y):
            feed_dict[placeholder] = value["scores"]

        loss_values = self._sess.run(self._loss_total, feed_dict=feed_dict)

        return loss_values

    def compute_losses(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute all loss components.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Dictionary of loss components.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Create feed_dict
        feed_dict = {self.images: x_preprocessed}

        for (placeholder, value) in zip(self._groundtruth_boxes_list, y):
            feed_dict[placeholder] = value["boxes"]

        for (placeholder, value) in zip(self._groundtruth_classes_list, y):
            feed_dict[placeholder] = value["labels"]

        for (placeholder, value) in zip(self._groundtruth_weights_list, y):
            feed_dict[placeholder] = value["scores"]

        # Get the losses graph
        if not hasattr(self, "_losses_dict"):
            self._losses_dict = dict()
            for loss_name in self.attack_losses:
                self._losses_dict[loss_name] = self._losses[loss_name]

        losses = dict()
        for loss_name in self.attack_losses:
            loss_value = self._sess.run(self._losses_dict[loss_name], feed_dict=feed_dict)
            losses[loss_name] = loss_value

        return losses
