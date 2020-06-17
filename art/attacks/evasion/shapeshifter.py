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
This module implements ShapeShifter, a robust physical adversarial attack on Faster R-CNN object detector.

| Paper link: https://arxiv.org/abs/1804.05810
"""
import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.tensorflow import TensorFlowEstimator
from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.utils import Deprecated, deprecated_keyword_arg

if TYPE_CHECKING:
    from object_detection.meta_architectures.faster_rcnn_meta_arch import FasterRCNNMetaArch
    from tensorflow.python.framework.ops import Tensor
    from tensorflow.python.client.session import Session

logger = logging.getLogger(__name__)


class ShapeShifter(EvasionAttack):
    """
    Implementation of the ShapeShifter attack. This is a robust physical adversarial attack on Faster R-CNN object
    detector and is developed in TensorFlow.

    | Paper link: https://arxiv.org/abs/1804.05810
    """

    attack_params = EvasionAttack.attack_params + [
        "random_transform",
        "box_classifier_weight",
        "box_localizer_weight",
        "rpn_classifier_weight",
        "rpn_localizer_weight",
        "box_iou_threshold",
        "box_victim_weight",
        "box_target_weight",
        "box_victim_cw_weight",
        "box_victim_cw_confidence",
        "box_target_cw_weight",
        "box_target_cw_confidence",
        "rpn_iou_threshold",
        "rpn_background_weight",
        "rpn_foreground_weight",
        "rpn_cw_weight",
        "rpn_cw_confidence",
        "similarity_weight",
        "learning_rate",
        "optimizer",
        "momentum",
        "decay",
        "sign_gradients",
        "random_size",
        "batch_random_size"
    ]

    _estimator_requirements = (
        BaseEstimator,
        LossGradientsMixin,
        NeuralNetworkMixin,
        ObjectDetectorMixin,
        TensorFlowEstimator,
        TensorFlowFasterRCNN
    )

    def __init__(
        self,
        estimator: TensorFlowFasterRCNN,
        random_transform: Tensor = tf.identity,
        box_classifier_weight: float = 1.0,
        box_localizer_weight: float = 2.0,
        rpn_classifier_weight: float = 1.0,
        rpn_localizer_weight: float = 2.0,
        box_iou_threshold: float = 0.5,
        box_victim_weight: float = 0.0,
        box_target_weight: float = 0.0,
        box_victim_cw_weight: float = 0.0,
        box_victim_cw_confidence: float = 0.0,
        box_target_cw_weight: float = 0.0,
        box_target_cw_confidence: float = 0.0,
        rpn_iou_threshold: float = 0.5,
        rpn_background_weight: float = 0.0,
        rpn_foreground_weight: float = 0.0,
        rpn_cw_weight: float = 0.0,
        rpn_cw_confidence: float = 0.0,
        similarity_weight: float = 0.0,
        learning_rate: float = 1.0,
        optimizer: str = 'GradientDescentOptimizer',
        momentum: float = 0.0,
        decay: float = 0.0,
        sign_gradients: bool = False,
        random_size: int = 1,
        batch_random_size: int = 1
    ):
        """
        Create an instance of the :class:`.ShapeShifter`.

        :param estimator: A trained object detector.
        :param random_transform: A TensorFlow function applies random transformations to images.
        :param box_classifier_weight: Weight of box classifier loss.
        :param box_localizer_weight: Weight of box localizer loss.
        :param rpn_classifier_weight: Weight of RPN classifier loss.
        :param rpn_localizer_weight: Weight of RPN localizer loss.
        :param box_iou_threshold: Box intersection over union threshold.
        :param box_victim_weight: Weight of box victim loss.
        :param box_target_weight: Weight of box target loss.
        :param box_victim_cw_weight: Weight of box victim CW loss.
        :param box_victim_cw_confidence: Confidence of box victim CW loss.
        :param box_target_cw_weight: Weight of box target CW loss.
        :param box_target_cw_confidence: Confidence of box target CW loss.
        :param rpn_iou_threshold: RPN intersection over union threshold.
        :param rpn_background_weight: Weight of RPN background loss.
        :param rpn_foreground_weight: Weight of RPN foreground loss.
        :param rpn_cw_weight: Weight of RPN CW loss.
        :param rpn_cw_confidence: Confidence of RPN CW loss.
        :param similarity_weight: Weight of similarity loss.
        :param learning_rate: Learning rate.
        :param optimizer: Optimizer including one of the following choices: `GradientDescentOptimizer`,
               `MomentumOptimizer`, `RMSPropOptimizer`, `AdamOptimizer`.
        :param momentum: Momentum for `RMSPropOptimizer`, `MomentumOptimizer`.
        :param decay: Learning rate decay for `RMSPropOptimizer`.
        :param sign_gradients: Whether to use the sign of gradients for optimization.
        :param random_size: Random sample size.
        :param batch_random_size: Batch size of random samples.
        """
        super(ShapeShifter, self).__init__(estimator=estimator)

        # Set attack attributes
        self.random_transform = random_transform
        self.box_classifier_weight = box_classifier_weight
        self.box_localizer_weight = box_localizer_weight
        self.rpn_classifier_weight = rpn_classifier_weight
        self.rpn_localizer_weight = rpn_localizer_weight
        self.box_iou_threshold = box_iou_threshold
        self.box_victim_weight = box_victim_weight
        self.box_target_weight = box_target_weight
        self.box_victim_cw_weight = box_victim_cw_weight
        self.box_victim_cw_confidence = box_victim_cw_confidence
        self.box_target_cw_weight = box_target_cw_weight
        self.box_target_cw_confidence = box_target_cw_confidence
        self.rpn_iou_threshold = rpn_iou_threshold
        self.rpn_background_weight = rpn_background_weight
        self.rpn_foreground_weight = rpn_foreground_weight
        self.rpn_cw_weight = rpn_cw_weight
        self.rpn_cw_confidence = rpn_cw_confidence
        self.similarity_weight = similarity_weight
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.decay = decay
        self.sign_gradients = sign_gradients
        self.random_size = random_size
        self.batch_random_size = batch_random_size

        # Check validity of attack attributes
        self._check_params()




    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """

        if not isinstance(self.rpn_cw_confidence, float):
            raise ValueError("The confidence of RPN CW loss must be of type float.")
        if not self.rpn_cw_confidence >= 0.0:
            raise ValueError("The confidence of RPN CW loss must be greater than or equal to 0.0.")

        if not isinstance(self.similarity_weight, float):
            raise ValueError("The weight of similarity loss must be of type float.")
        if not self.similarity_weight >= 0.0:
            raise ValueError("The weight of similarity loss must be greater than or equal to 0.0.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if self.optimizer not in ['RMSPropOptimizer', 'MomentumOptimizer', 'GradientDescentOptimizer', 'AdamOptimizer']:
            raise ValueError(
                "Optimizer only includes one of the following choices: `GradientDescentOptimizer`, "
                "`MomentumOptimizer`, `RMSPropOptimizer`, `AdamOptimizer`."
            )

        if self.optimizer in ['RMSPropOptimizer', 'MomentumOptimizer']:
            if not isinstance(self.momentum, float):
                raise ValueError("The momentum must be of type float.")
            if not self.momentum > 0.0:
                raise ValueError("The momentum must be greater than 0.0.")

        if self.optimizer == 'RMSPropOptimizer':
            if not isinstance(self.decay, float):
                raise ValueError("The learning rate decay must be of type float.")
            if not self.decay > 0.0:
                raise ValueError("The learning rate decay must be greater than 0.0.")
            if not self.decay < 1.0:
                raise ValueError("The learning rate decay must be smaller than 1.0.")

        if not isinstance(self.sign_gradients, bool):
            raise ValueError(
                "The choice of whether to use the sign of gradients for the optimization must be of type bool."
            )

        if not isinstance(self.random_size, int):
            raise ValueError("The random sample size must be of type int.")
        if not self.random_size > 0:
            raise ValueError("The random sample size must be greater than 0.")

        if not isinstance(self.batch_random_size, int):
            raise ValueError("The batch size of random samples must be of type int.")
        if not self.batch_size > 0:
            raise ValueError("The batch size of random samples must be greater than 0.")







    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: Sample images.
        :type x: `np.ndarray`
        :param y: Target labels for object detector.
        :type y: `np.ndarray`
        :return: Adversarial patch.
        :rtype: `np.ndarray`
        """
        assert x.ndim == 4, "The adversarial patch can only be applied to images."








        return self._patch


