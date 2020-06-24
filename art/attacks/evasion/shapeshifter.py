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
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.tensorflow import TensorFlowEstimator
from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
from art.estimators.object_detection.object_detector import ObjectDetectorMixin

if TYPE_CHECKING:
    from collections import Callable

    from tensorflow.python.framework.ops import Tensor
    from tensorflow.python.training.optimizer import Optimizer

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
        "max_iter",
        "texture_as_input",
        "use_spectral",
        "soft_clip"
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
        random_transform: Callable,
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
        random_size: int = 10,
        max_iter: int = 10,
        texture_as_input: bool = False,
        use_spectral: bool = True,
        soft_clip: bool = False
    ):
        """
        Create an instance of the :class:`.ShapeShifter`.

        :param estimator: A trained object detector.
        :param random_transform: A function applies random transformations to images/textures.
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
        :param max_iter: Maximum number of iterations.
        :param texture_as_input: Whether textures are used as inputs instead of images.
        :param use_spectral: Whether to use spectral with textures.
        :param soft_clip: Whether to apply soft clipping on textures.
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
        self.max_iter = max_iter
        self.texture_as_input = texture_as_input
        self.use_spectral = use_spectral
        self.soft_clip = soft_clip
        self.graph_available: bool = False

        # Check validity of attack attributes
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[Dict[str, List[Tensor]]] = None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: Sample image.
        :param y: Target labels for object detector (not used).
        :param target_class: Target class.
        :type target_class: int
        :param victim_class: Victim class.
        :type victim_class: int
        :param custom_loss: Custom loss function from users.
        :type custom_loss: Tensor
        :param rendering_function: A rendering function to use textures as input.
        :type rendering_function: Callable
        :return: Adversarial image/texture.
        :rtype: `np.ndarray`
        """
        # Check input shape
        assert x.ndim == 4, "The ShapeShifter attack can only be applied to images."
        assert x.shape[0] == 1, "The ShapeShifter attack can only be applied to one image."
        assert x.shape[1:] == self.estimator.input_shape

        # Check whether users have a custom loss
        custom_loss = kwargs.get("custom_loss")

        # Build the TensorFlow graph
        if not self.graph_available:
            self.graph_available = True

            if self.texture_as_input:
                # Check whether users provide a rendering function
                rendering_function = kwargs.get("rendering_function")
                if rendering_function is None:
                    raise ValueError("Need a rendering function to use textures as input.")

                # Build the TensorFlow graph
                (
                    project_texture_op,
                    current_image_assign_to_input_image_op,
                    accumulated_gradients_op,
                    final_attack_optimization_op,
                    current_variable,
                    current_value
                ) = self._build_graph(
                    initial_shape=x.shape, custom_loss=custom_loss, rendering_function=rendering_function
                )

            else:
                (
                    project_texture_op,
                    current_image_assign_to_input_image_op,
                    accumulated_gradients_op,
                    final_attack_optimization_op,
                    current_variable,
                    current_value
                ) = self._build_graph(initial_shape=x.shape, custom_loss=custom_loss)

        # Do attack
        result = self._attack_training(
            project_texture_op=project_texture_op,
            current_image_assign_to_input_image_op=current_image_assign_to_input_image_op,
            accumulated_gradients_op=accumulated_gradients_op,
            final_attack_optimization_op=final_attack_optimization_op,
            current_variable=current_variable,
            current_value=current_value
        )

        return result

    def _attack_training(
        self,
        project_texture_op: Tensor,
        current_image_assign_to_input_image_op: Tensor,
        accumulated_gradients_op: Tensor,
        final_attack_optimization_op: Tensor,
        current_variable: Tensor,
        current_value: Tensor
    ):
        """
        Do attack optimization.

        :param project_texture_op:
        :param current_image_assign_to_input_image_op:
        :param accumulated_gradients_op:
        :param final_attack_optimization_op:
        :param current_value:
        :param current_variable:

        :return:
        """
        return 0


    def _build_graph(
        self,
        initial_shape: List,
        custom_loss: Optional[Tensor] = None,
        rendering_function: Optional[Callable] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Build the TensorFlow graph for the attack.

        :param initial_shape: Image/texture shape.
        :param custom_loss: Custom loss function from users.
        :param rendering_function: A rendering function to use textures as input.
        :return: A tuple of tensors.
        """
        # Create a placeholder to pass input image/texture
        initial_input = tf.placeholder(dtype=tf.float32, shape=initial_shape, name='initial_input')

        # Create a placeholder to pass input image/texture mask
        mask_input = tf.placeholder(dtype=tf.float32, shape=initial_shape, name='mask_input')

        # Create adversarial image
        if self.texture_as_input:
            # Create texture variable
            if self.use_spectral:
                initial_value = np.zeros(
                    (2, initial_shape[0], initial_shape[3], initial_shape[1], int(np.ceil(initial_shape[2] / 2) + 1))
                )

                current_texture_variable = tf.Variable(
                    initial_value=initial_value, dtype=tf.float32, name='current_texture_variable'
                )

                current_texture = current_texture_variable
                current_texture = tf.complex(current_texture[0], current_texture[1])
                current_texture = tf.map_fn(tf.spectral.irfft2d, current_texture, dtype=tf.float32)
                current_texture = tf.transpose(current_texture, (0, 2, 3, 1))

            else:
                initial_value = np.zeros((initial_shape[0], initial_shape[1], initial_shape[2], initial_shape[3]))

                current_texture_variable = tf.Variable(
                    initial_value=initial_value, dtype=tf.float32, name='current_texture_variable'
                )

                current_texture = current_texture_variable

            # Invert texture for projection
            project_texture = initial_input * (1.0 - mask_input) + current_texture * mask_input

            if self.soft_clip:
                project_texture = tf.nn.sigmoid(project_texture)
            else:
                project_texture = tf.clip_by_value(project_texture, 0., 1.)

            if self.use_spectral:
                project_texture = tf.transpose(project_texture, (0, 3, 1, 2))
                project_texture = tf.map_fn(tf.spectral.rfft2d, project_texture, dtype=tf.complex64)
                project_texture = tf.stack([tf.real(project_texture), tf.imag(project_texture)])

            # Update texture variable
            project_texture_op = tf.assign(current_texture_variable, project_texture, name='project_texture_op')

            # Create a placeholder to pass the background
            background_phd = tf.placeholder(
                dtype=tf.float32, shape=[initial_shape[0], None, None, 3], name='background_phd'
            )

            # Create a placeholder to pass the image frame
            image_frame_phd = tf.placeholder(
                dtype=tf.float32, shape=[initial_shape[0], None, None, 4], name='image_frame_phd'
            )

            # Create adversarial image
            current_image = rendering_function(background_phd, image_frame_phd, current_texture)

        else:
            # Create image variable
            current_image_variable = tf.Variable(
                initial_value=np.zeros(initial_input.shape.as_list()), dtype=tf.float32, name='current_image_variable'
            )

            current_image = (tf.tanh(current_image_variable) + 1) / 2
            current_image = tf.identity(current_image, name='current_image')

        # # Generate random transformed images
        # # Define the stop condition
        # def stop_condition(i, _):
        #     return tf.less(i, self.batch_random_size)
        #
        # # Random transformation of the image
        # def main_body_loop(i, batch):
        #     transformed_image = self.random_transform(current_image)
        #     batch = tf.cond(
        #         tf.equal(i, 0),
        #         lambda: tf.concat([[transformed_image]], axis=0),
        #         lambda: tf.concat([batch, [transformed_image]], axis=0)
        #     )
        #     return i + 1, batch
        #
        # _, batch = tf.while_loop(
        #     cond=stop_condition,
        #     body=main_body_loop,
        #     loop_vars=[tf.zeros([]), None],
        #     back_prop=True,
        #     name='generate_random_transformed_images'
        # )
        #
        # # Assign batch to the input of the object detector
        # batch_assign_to_input_images = tf.assign(
        #     ref=self.estimator.input_images, value=batch, name='batch_assign_to_input_images'
        # )

        # Assign current image to the input of the object detector
        current_image_assign_to_input_image_op = tf.assign(
            ref=self.estimator.input_images, value=current_image, name='current_image_assign_to_input_image_op'
        )

        # Create attack loss
        if self.texture_as_input:
            total_loss = self._create_attack_loss(
                custom_loss=custom_loss, initial_input=initial_input, current_value=current_texture
            )
        else:
            total_loss = self._create_attack_loss(
                custom_loss=custom_loss, initial_input=initial_input, current_value=current_image
            )

        # Create optimizer
        optimizer = self._create_optimizer()

        # Create gradients
        gradients = optimizer.compute_gradients(total_loss, var_list=[current_image])[0][0]

        # Create variables to store gradients
        if self.texture_as_input:
            sum_gradients = tf.Variable(
                initial_value=np.zeros(current_texture_variable.shape.as_list()),
                trainable=False,
                name='sum_gradients',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES]
            )

        else:
            sum_gradients = tf.Variable(
                initial_value=np.zeros(current_image_variable.shape.as_list()),
                trainable=False,
                name='sum_gradients',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES]
            )

        num_gradients = tf.Variable(
            initial_value=0.,
            trainable=False,
            name='count_gradients',
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES]
        )

        # Accumulate gradients
        accumulated_sum_gradients = tf.assign_add(sum_gradients, gradients)
        accumulated_num_gradients = tf.assign_add(num_gradients, 1.)

        # Final gradients
        final_gradients = tf.div(
            accumulated_sum_gradients, tf.maximum(accumulated_num_gradients, 1.), name='final_gradients'
        )

        if self.sign_gradients:
            final_gradients = tf.sign(final_gradients)

        # Create accumulated gradients operator
        accumulated_gradients_op = tf.group(
            [accumulated_sum_gradients, accumulated_num_gradients], name='accumulated_gradients_op'
        )

        # Create final attack optimization operator
        final_attack_optimization_op = optimizer.apply_gradients(
            grads_and_vars=[(final_gradients, current_image)], name='final_attack_optimization_op'
        )

        if self.texture_as_input:
            return (
                project_texture_op,
                current_image_assign_to_input_image_op,
                accumulated_gradients_op,
                final_attack_optimization_op,
                current_texture_variable,
                current_texture
            )

        else:
            return (
                project_texture_op,
                current_image_assign_to_input_image_op,
                accumulated_gradients_op,
                final_attack_optimization_op,
                current_image_variable,
                current_image
            )

    def _create_optimizer(self) -> Optimizer:
        """
        Create an optimizer of this attack.

        :return: Attack optimizer.
        """
        # Create placeholder for learning rate
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

        # Create placeholder for momentum
        if self.optimizer in ['RMSPropOptimizer', 'MomentumOptimizer']:
            momentum = tf.placeholder(dtype=tf.float32, shape=[], name='momentum')

        # Create placeholder for decay
        if self.optimizer == 'RMSPropOptimizer':
            decay = tf.placeholder(dtype=tf.float32, shape=[], name='decay')

        # Create optimizer
        if self.optimizer == 'GradientDescentOptimizer':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif self.optimizer == 'MomentumOptimizer':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        elif self.optimizer == 'RMSPropOptimizer':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, decay=decay)
        elif self.optimizer == 'AdamOptimizer':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError("Unknown optimizer.")

        return optimizer

    def _create_attack_loss(
        self,
        custom_loss: Optional[Tensor] = None,
        initial_input: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None
    ) -> Tensor:
        """
        Create the loss tensor of this attack.

        :param custom_loss: Custom loss function from users.
        :param initial_input: Initial input.
        :param current_value: Current image/texture.
        :return: Attack loss tensor.
        """
        # Compute faster rcnn loss
        partial_faster_rcnn_loss = self._create_faster_rcnn_loss()

        # Compute box loss
        partial_box_loss = self._create_box_loss()

        # Compute RPN loss
        partial_rpn_loss = self._create_rpn_loss()

        # Compute similarity loss
        weight_similarity_loss = self._create_similarity_loss(initial_input=initial_input, current_value=current_value)

        # Compute total loss
        if custom_loss is not None:
            total_loss = tf.add_n(
                [
                    partial_faster_rcnn_loss,
                    partial_box_loss,
                    partial_rpn_loss,
                    weight_similarity_loss,
                    custom_loss
                ],
                name='total_loss'
            )

        else:
            total_loss = tf.add_n(
                [
                    partial_faster_rcnn_loss,
                    partial_box_loss,
                    partial_rpn_loss,
                    weight_similarity_loss
                ],
                name='total_loss'
            )

        return total_loss

    def _create_faster_rcnn_loss(self) -> Tensor:
        """
        Create the partial loss tensor of this attack from losses of the object detector.

        :return: Attack partial loss tensor.
        """
        # Compute RPN classifier loss
        rpn_classifier_weight = tf.placeholder(dtype=tf.float32, shape=[], name='rpn_classifier_weight')

        rpn_classifier_loss = self.estimator.losses['Loss/RPNLoss/objectness_loss']
        weight_rpn_classifier_loss = tf.multiply(
            x=rpn_classifier_loss, y=rpn_classifier_weight, name='weight_rpn_classifier_loss'
        )

        # Compute RPN localizer loss
        rpn_localizer_weight = tf.placeholder(dtype=tf.float32, shape=[], name='rpn_localizer_weight')

        rpn_localizer_loss = self.estimator.losses['Loss/RPNLoss/localization_loss']
        weight_rpn_localizer_loss = tf.multiply(
            x=rpn_localizer_loss, y=rpn_localizer_weight, name='weight_rpn_localizer_loss'
        )

        # Compute box classifier loss
        box_classifier_weight = tf.placeholder(dtype=tf.float32, shape=[], name='box_classifier_weight')

        box_classifier_loss = self.estimator.losses['Loss/BoxClassifierLoss/classification_loss']
        weight_box_classifier_loss = tf.multiply(
            x=box_classifier_loss, y=box_classifier_weight, name='weight_box_classifier_loss'
        )

        # Compute box localizer loss
        box_localizer_weight = tf.placeholder(dtype=tf.float32, shape=[], name='box_localizer_weight')

        box_localizer_loss = self.estimator.losses['Loss/BoxClassifierLoss/localization_loss']
        weight_box_localizer_loss = tf.multiply(
            x=box_localizer_loss, y=box_localizer_weight, name='weight_box_localizer_loss'
        )

        # Compute partial loss
        partial_loss = tf.add_n(
            [
                weight_rpn_classifier_loss,
                weight_rpn_localizer_loss,
                weight_box_classifier_loss,
                weight_box_localizer_loss
            ],
            name='partial_faster_rcnn_loss'
        )

        return partial_loss

    def _create_box_loss(self) -> Tensor:
        """
        Create the partial loss tensor of this attack from box losses.

        :return: Attack partial loss tensor.
        """
        # Get default graph
        default_graph = tf.get_default_graph()

        # Compute box losses
        target_class_phd = tf.placeholder(dtype=tf.int32, shape=[], name='target_class_phd')
        victim_class_phd = tf.placeholder(dtype=tf.int32, shape=[], name='victim_class_phd')
        box_iou_threshold = tf.placeholder(dtype=tf.float32, shape=[], name='box_iou_threshold')

        # Ignore background class
        class_predictions_with_background = self.estimator.predictions['class_predictions_with_background']
        class_predictions_with_background = class_predictions_with_background[:, 1:]

        # Convert to 1-hot
        target_class_one_hot = tf.one_hot([target_class_phd - 1], class_predictions_with_background.shape[-1])
        victim_class_one_hot = tf.one_hot([victim_class_phd - 1], class_predictions_with_background.shape[-1])

        box_iou_tensor = default_graph.get_tensor_by_name('Loss/BoxClassifierLoss/Compare/IOU/Select:0')
        box_iou_tensor = tf.reshape(box_iou_tensor, (-1,))
        box_target = tf.cast(box_iou_tensor >= box_iou_threshold, dtype=tf.float32)

        # Compute box target loss
        box_target_weight = tf.placeholder(dtype=tf.float32, shape=[], name='box_target_weight')

        box_target_logit = class_predictions_with_background[:, target_class_phd - 1]
        box_target_loss = box_target_logit * box_target
        box_target_loss = -1 * tf.reduce_sum(box_target_loss)
        weight_box_target_loss = tf.multiply(
            x=box_target_loss, y=box_target_weight, name='weight_box_target_loss'
        )

        # Compute box victim loss
        box_victim_weight = tf.placeholder(dtype=tf.float32, shape=[], name='box_victim_weight')

        box_victim_logit = class_predictions_with_background[:, victim_class_phd - 1]
        box_victim_loss = box_victim_logit * box_target
        box_victim_loss = tf.reduce_sum(box_victim_loss)
        weight_box_victim_loss = tf.multiply(
            x=box_victim_loss, y=box_victim_weight, name='weight_box_victim_loss'
        )

        # Compute box target CW loss
        box_target_cw_weight = tf.placeholder(dtype=tf.float32, shape=[], name='box_target_cw_weight')
        box_target_cw_confidence = tf.placeholder(dtype=tf.float32, shape=[], name='box_target_cw_confidence')

        box_nontarget_logit = tf.reduce_max(
            class_predictions_with_background * (1 - target_class_one_hot) - 10000 * target_class_one_hot, axis=-1
        )
        box_target_cw_loss = tf.nn.relu(box_nontarget_logit - box_target_logit + box_target_cw_confidence)
        box_target_cw_loss = box_target_cw_loss * box_target
        box_target_cw_loss = tf.reduce_sum(box_target_cw_loss)
        weight_box_target_cw_loss = tf.multiply(
            x=box_target_cw_loss, y=box_target_cw_weight, name='weight_box_target_cw_loss'
        )

        # Compute box victim CW loss
        box_victim_cw_weight = tf.placeholder(dtype=tf.float32, shape=[], name='box_victim_cw_weight')
        box_victim_cw_confidence = tf.placeholder(dtype=tf.float32, shape=[], name='box_victim_cw_confidence')

        box_nonvictim_logit = tf.reduce_max(
            class_predictions_with_background * (1 - victim_class_one_hot) - 10000 * victim_class_one_hot, axis=-1
        )
        box_victim_cw_loss = tf.nn.relu(box_victim_logit - box_nonvictim_logit + box_victim_cw_confidence)
        box_victim_cw_loss = box_victim_cw_loss * box_target
        box_victim_cw_loss = tf.reduce_sum(box_victim_cw_loss)
        weight_box_victim_cw_loss = tf.multiply(
            x=box_victim_cw_loss, y=box_victim_cw_weight, name='weight_box_victim_cw_loss'
        )

        # Compute partial loss
        partial_loss = tf.add_n(
            [
                weight_box_target_loss,
                weight_box_victim_loss,
                weight_box_target_cw_loss,
                weight_box_victim_cw_loss
            ],
            name='partial_box_loss'
        )

        return partial_loss

    def _create_rpn_loss(self) -> Tensor:
        """
        Create the partial loss tensor of this attack from RPN losses.

        :return: Attack partial loss tensor.
        """
        # Get default graph
        default_graph = tf.get_default_graph()

        # Compute RPN losses
        rpn_iou_threshold = tf.placeholder(dtype=tf.float32, shape=[], name='rpn_iou_threshold')

        # RPN background
        rpn_objectness_predictions_with_background = self.estimator.predictions[
            'rpn_objectness_predictions_with_background'
        ]
        rpn_objectness_predictions_with_background = tf.reshape(
            rpn_objectness_predictions_with_background, (-1, rpn_objectness_predictions_with_background.shape[-1])
        )
        rpn_iou_tensor = default_graph.get_tensor_by_name('Loss/RPNLoss/Compare/IOU/Select:0')
        rpn_iou_tensor = tf.reshape(rpn_iou_tensor, (-1,))
        rpn_target = tf.cast(rpn_iou_tensor >= rpn_iou_threshold, dtype=tf.float32)

        # Compute RPN background loss
        rpn_background_weight = tf.placeholder(dtype=tf.float32, shape=[], name='rpn_background_weight')

        rpn_background_logit = rpn_objectness_predictions_with_background[:, 0]
        rpn_background_loss = rpn_background_logit * rpn_target
        rpn_background_loss = -1 * tf.reduce_sum(rpn_background_loss)
        weight_rpn_background_loss = tf.multiply(
            x=rpn_background_loss, y=rpn_background_weight, name='weight_rpn_background_loss'
        )

        # Compute RPN foreground loss
        rpn_foreground_weight = tf.placeholder(dtype=tf.float32, shape=[], name='rpn_foreground_weight')

        rpn_foreground_logit = rpn_objectness_predictions_with_background[:, 1]
        rpn_foreground_loss = rpn_foreground_logit * rpn_target
        rpn_foreground_loss = tf.reduce_sum(rpn_foreground_loss)
        weight_rpn_foreground_loss = tf.multiply(
            x=rpn_foreground_loss, y=rpn_foreground_weight, name='weight_rpn_foreground_loss'
        )

        # Compute RPN CW loss
        rpn_cw_weight = tf.placeholder(dtype=tf.float32, shape=[], name='rpn_cw_weight')
        rpn_cw_confidence = tf.placeholder(dtype=tf.float32, shape=[], name='rpn_cw_confidence')

        rpn_cw_loss = tf.nn.relu(rpn_foreground_logit - rpn_background_logit + rpn_cw_confidence)
        rpn_cw_loss = rpn_cw_loss * rpn_target
        rpn_cw_loss = tf.reduce_sum(rpn_cw_loss)
        weight_rpn_cw_loss = tf.multiply(
            x=rpn_cw_loss, y=rpn_cw_weight, name='weight_rpn_cw_loss'
        )

        # Compute partial loss
        partial_loss = tf.add_n(
            [
                weight_rpn_background_loss,
                weight_rpn_foreground_loss,
                weight_rpn_cw_loss,
            ],
            name='partial_rpn_loss'
        )

        return partial_loss

    @staticmethod
    def _create_similarity_loss(
        initial_input: Optional[Tensor] = None,
        current_value: Optional[Tensor] = None
    ) -> Tensor:
        """
        Create the partial loss tensor of this attack from the similarity loss.

        :param initial_input: Initial input.
        :param current_value: Current image/texture.
        :return: Attack partial loss tensor.
        """
        # Create a placeholder for the similarity weight
        similarity_weight = tf.placeholder(dtype=tf.float32, shape=[], name='similarity_weight')

        # Compute similarity loss
        similarity_loss = tf.nn.l2_loss(initial_input - current_value)
        weight_similarity_loss = tf.multiply(
            x=similarity_loss, y=similarity_weight, name='weight_similarity_loss'
        )

        return weight_similarity_loss

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if not isinstance(self.random_transform, Callable):
            raise ValueError("The applied random transformation function must be of type Callable.")

        if not isinstance(self.box_classifier_weight, float):
            raise ValueError("The weight of box classifier loss must be of type float.")
        if not self.box_classifier_weight >= 0.0:
            raise ValueError("The weight of box classifier loss must be greater than or equal to 0.0.")

        if not isinstance(self.box_localizer_weight, float):
            raise ValueError("The weight of box localizer loss must be of type float.")
        if not self.box_localizer_weight >= 0.0:
            raise ValueError("The weight of box localizer loss must be greater than or equal to 0.0.")

        if not isinstance(self.rpn_classifier_weight, float):
            raise ValueError("The weight of RPN classifier loss must be of type float.")
        if not self.rpn_classifier_weight >= 0.0:
            raise ValueError("The weight of RPN classifier loss must be greater than or equal to 0.0.")

        if not isinstance(self.rpn_localizer_weight, float):
            raise ValueError("The weight of RPN localizer loss must be of type float.")
        if not self.rpn_localizer_weight >= 0.0:
            raise ValueError("The weight of RPN localizer loss must be greater than or equal to 0.0.")

        if not isinstance(self.box_iou_threshold, float):
            raise ValueError("The box intersection over union threshold must be of type float.")
        if not self.box_iou_threshold >= 0.0:
            raise ValueError("The box intersection over union threshold must be greater than or equal to 0.0.")

        if not isinstance(self.box_victim_weight, float):
            raise ValueError("The weight of box victim loss must be of type float.")
        if not self.box_victim_weight >= 0.0:
            raise ValueError("The weight of box victim loss must be greater than or equal to 0.0.")

        if not isinstance(self.box_target_weight, float):
            raise ValueError("The weight of box target loss must be of type float.")
        if not self.box_target_weight >= 0.0:
            raise ValueError("The weight of box target loss must be greater than or equal to 0.0.")

        if not isinstance(self.box_victim_cw_weight, float):
            raise ValueError("The weight of box victim CW loss must be of type float.")
        if not self.box_victim_cw_weight >= 0.0:
            raise ValueError("The weight of box victim CW loss must be greater than or equal to 0.0.")

        if not isinstance(self.box_victim_cw_confidence, float):
            raise ValueError("The confidence of box victim CW loss must be of type float.")
        if not self.box_victim_cw_confidence >= 0.0:
            raise ValueError("The confidence of box victim CW loss must be greater than or equal to 0.0.")

        if not isinstance(self.box_target_cw_weight, float):
            raise ValueError("The weight of box target CW loss must be of type float.")
        if not self.box_target_cw_weight >= 0.0:
            raise ValueError("The weight of box target CW loss must be greater than or equal to 0.0.")

        if not isinstance(self.box_target_cw_confidence, float):
            raise ValueError("The confidence of box target CW loss must be of type float.")
        if not self.box_target_cw_confidence >= 0.0:
            raise ValueError("The confidence of box target CW loss must be greater than or equal to 0.0.")

        if not isinstance(self.rpn_iou_threshold, float):
            raise ValueError("The RPN intersection over union threshold must be of type float.")
        if not self.rpn_iou_threshold >= 0.0:
            raise ValueError("The RPN intersection over union threshold must be greater than or equal to 0.0.")

        if not isinstance(self.rpn_background_weight, float):
            raise ValueError("The weight of RPN background loss must be of type float.")
        if not self.rpn_background_weight >= 0.0:
            raise ValueError("The weight of RPN background loss must be greater than or equal to 0.0.")

        if not isinstance(self.rpn_foreground_weight, float):
            raise ValueError("The weight of RPN foreground loss must be of type float.")
        if not self.rpn_foreground_weight >= 0.0:
            raise ValueError("The weight of RPN foreground loss must be greater than or equal to 0.0.")

        if not isinstance(self.rpn_cw_weight, float):
            raise ValueError("The weight of RPN CW loss must be of type float.")
        if not self.rpn_cw_weight >= 0.0:
            raise ValueError("The weight of RPN CW loss must be greater than or equal to 0.0.")

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

        if not isinstance(self.max_iter, int):
            raise ValueError("The maximum number of iterations must be of type int.")
        if not self.max_iter > 0:
            raise ValueError("The maximum number of iterations must be greater than 0.")

        if not isinstance(self.texture_as_input, bool):
            raise ValueError(
                "The choice of whether textures are used as inputs instead of images must be of type bool."
            )

        if not isinstance(self.use_spectral, bool):
            raise ValueError(
                "The choice of whether to use spectral with textures must be of type bool."
            )

        if not isinstance(self.soft_clip, bool):
            raise ValueError(
                "The choice of whether to apply soft clipping on textures must be of type bool."
            )
