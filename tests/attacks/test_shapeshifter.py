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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest
import importlib

import tensorflow as tf
import numpy as np

from tests.utils import TestBase, master_seed

object_detection_spec = importlib.util.find_spec("object_detection")
object_detection_found = object_detection_spec is not None

logger = logging.getLogger(__name__)


@unittest.skipIf(
    not object_detection_found,
    reason="Skip unittests if object detection module is not found because of pre-trained model.",
)
@unittest.skipIf(
    tf.__version__[0] == "2" or (tf.__version__[0] == "1" and tf.__version__.split(".")[1] != "15"),
    reason="Skip unittests if not TensorFlow v1.15 because of pre-trained model.",
)
class TestShapeShifter(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        cls.n_test = 10
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def test_image_as_input(self):
        self._test_image_as_input(True)
        self._test_image_as_input(False)

    def _test_image_as_input(self, sign_gradients):
        # We must start a new graph
        tf.reset_default_graph()

        # Only import if object detection module is available
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
        from art.attacks.evasion.shapeshifter import ShapeShifter

        # Define object detector
        images = tf.Variable(initial_value=np.zeros([1, 28, 28, 1]), dtype=tf.float32)
        obj_dec = TensorFlowFasterRCNN(images=images)

        # Create labels
        result = obj_dec.predict(self.x_test_mnist[:1].astype(np.float32))

        groundtruth_boxes_list = [result["detection_boxes"][i] for i in range(1)]
        groundtruth_classes_list = [result["detection_classes"][i] for i in range(1)]
        groundtruth_weights_list = [np.ones_like(r) for r in groundtruth_classes_list]

        y = {
            "groundtruth_boxes_list": groundtruth_boxes_list,
            "groundtruth_classes_list": groundtruth_classes_list,
            "groundtruth_weights_list": groundtruth_weights_list,
        }

        # Define attack
        attack = ShapeShifter(
            estimator=obj_dec,
            random_transform=lambda x: x + 1e-10,
            box_classifier_weight=1.0,
            box_localizer_weight=1.0,
            rpn_classifier_weight=1.0,
            rpn_localizer_weight=1.0,
            box_iou_threshold=0.3,
            box_victim_weight=1.0,
            box_target_weight=1.0,
            box_victim_cw_weight=1.0,
            box_victim_cw_confidence=1.0,
            box_target_cw_weight=1.0,
            box_target_cw_confidence=1.0,
            rpn_iou_threshold=0.3,
            rpn_background_weight=1.0,
            rpn_foreground_weight=1.0,
            rpn_cw_weight=1.0,
            rpn_cw_confidence=1.0,
            similarity_weight=1.0,
            learning_rate=0.1,
            optimizer="RMSPropOptimizer",
            momentum=0.01,
            decay=0.01,
            sign_gradients=sign_gradients,
            random_size=2,
            max_iter=2,
            texture_as_input=False,
            use_spectral=False,
            soft_clip=True,
        )

        # Targeted attack
        adv_x = attack.generate(x=self.x_test_mnist[:1].astype(np.float32), label=y, target_class=2, victim_class=5)

        self.assertTrue(adv_x.shape == (1, 28, 28, 1))
        self.assertTrue((adv_x >= 0).all())
        self.assertTrue((adv_x <= 1).all())

        # Untargeted attack
        adv_x = attack.generate(x=self.x_test_mnist[:1].astype(np.float32), label=y, victim_class=8)

        self.assertTrue(adv_x.shape == (1, 28, 28, 1))
        self.assertTrue((adv_x >= 0).all())
        self.assertTrue((adv_x <= 1).all())

    def test_texture_as_input(self):
        self._test_texture_as_input(True, True, True)
        self._test_texture_as_input(True, True, False)
        self._test_texture_as_input(True, False, True)
        self._test_texture_as_input(True, False, False)
        self._test_texture_as_input(False, True, True)
        self._test_texture_as_input(False, True, False)
        self._test_texture_as_input(False, False, True)
        self._test_texture_as_input(False, False, False)

    def _test_texture_as_input(self, sign_gradients, use_spectral, soft_clip):
        # We must start a new graph
        tf.reset_default_graph()

        # Only import if object detection module is available
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
        from art.attacks.evasion.shapeshifter import ShapeShifter

        # Define object detector
        images = tf.Variable(initial_value=np.zeros([1, 28, 28, 1]), dtype=tf.float32)
        obj_dec = TensorFlowFasterRCNN(images=images)

        # Create labels
        result = obj_dec.predict(self.x_test_mnist[:1].astype(np.float32))

        groundtruth_boxes_list = [result["detection_boxes"][i] for i in range(1)]
        groundtruth_classes_list = [result["detection_classes"][i] for i in range(1)]
        groundtruth_weights_list = [np.ones_like(r) for r in groundtruth_classes_list]

        y = {
            "groundtruth_boxes_list": groundtruth_boxes_list,
            "groundtruth_classes_list": groundtruth_classes_list,
            "groundtruth_weights_list": groundtruth_weights_list,
        }

        # Define random transform
        def random_transform(x):
            background = np.random.rand(*x.shape)
            image_frame = np.random.rand(*(list(x.shape[:-1]) + [4]))

            y_ = y.copy()
            y_["groundtruth_boxes_list"][0] = y_["groundtruth_boxes_list"][0] + np.random.rand()
            y_["groundtruth_weights_list"][0] = y_["groundtruth_weights_list"][0] + np.random.rand()

            return background, image_frame, y_

        # Define attack
        attack = ShapeShifter(
            estimator=obj_dec,
            random_transform=random_transform,
            box_classifier_weight=1.0,
            box_localizer_weight=1.0,
            rpn_classifier_weight=1.0,
            rpn_localizer_weight=1.0,
            box_iou_threshold=0.3,
            box_victim_weight=1.0,
            box_target_weight=1.0,
            box_victim_cw_weight=1.0,
            box_victim_cw_confidence=1.0,
            box_target_cw_weight=1.0,
            box_target_cw_confidence=1.0,
            rpn_iou_threshold=0.3,
            rpn_background_weight=1.0,
            rpn_foreground_weight=1.0,
            rpn_cw_weight=1.0,
            rpn_cw_confidence=1.0,
            similarity_weight=1.0,
            learning_rate=0.1,
            optimizer="MomentumOptimizer",
            momentum=0.01,
            decay=0.01,
            sign_gradients=sign_gradients,
            random_size=2,
            max_iter=2,
            texture_as_input=True,
            use_spectral=use_spectral,
            soft_clip=soft_clip,
        )

        # Define rendering function
        def rendering_function(background_phd, image_frame_phd, current_texture):
            current_image = background_phd + current_texture
            current_image = tf.clip_by_value(current_image, 0, 1)

            return current_image

        # Targeted attack
        adv_x = attack.generate(
            x=self.x_test_mnist[:1].astype(np.float32),
            label=y,
            target_class=2,
            victim_class=5,
            rendering_function=rendering_function,
        )

        self.assertTrue(adv_x.shape == (1, 28, 28, 1))

        # Untargeted attack
        adv_x = attack.generate(
            x=self.x_test_mnist[:1].astype(np.float32),
            label=y,
            target_class=8,
            victim_class=8,
            rendering_function=rendering_function,
        )

        self.assertTrue(adv_x.shape == (1, 28, 28, 1))


if __name__ == "__main__":
    unittest.main()
