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

import numpy as np
import tensorflow as tf

from tests.utils import TestBase, master_seed

object_detection_spec = importlib.util.find_spec("object_detection")
object_detection_found = object_detection_spec is not None

logger = logging.getLogger(__name__)


@unittest.skipIf(
     not object_detection_found,
     reason="Skip unittests if object detection module is not found because of pre-trained model."
)
@unittest.skipIf(
    tf.__version__[0] == "2" or (tf.__version__[0] == "1" and tf.__version__.split(".")[1] != "15"),
    reason="Skip unittests if not TensorFlow v1.15 because of pre-trained model.",
)
class TestTensorFlowFasterRCNN(TestBase):
    """
    This class tests the TensorFlowFasterRCNN object detector.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        cls.n_test = 10
        cls.x_test_mnist = cls.x_test_mnist[0: cls.n_test]

    def setUp(self):
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN

        images = tf.placeholder(tf.float32, shape=[2, 28, 28, 1])
        self.obj_dec = TensorFlowFasterRCNN(images=images)

    def test_predict(self):
        result = self.obj_dec.predict(self.x_test_mnist)

        self.assertTrue(list(result.keys()) == [
            'detection_boxes',
            'detection_scores',
            'detection_classes',
            'detection_multiclass_scores',
            'detection_anchor_indices',
            'num_detections',
            'raw_detection_boxes',
            'raw_detection_scores'
        ])

        self.assertTrue(result['detection_boxes'].shape == (10, 300, 4))
        expected_detection_boxes = np.asarray([0.65566427, 0., 1., 0.9642794])
        np.testing.assert_array_almost_equal(result['detection_boxes'][0, 2, :], expected_detection_boxes, decimal=6)

        self.assertTrue(result['detection_scores'].shape == (10, 300))
        expected_detection_scores = np.asarray([
            6.02739106e-04,
            3.72770795e-04,
            2.96768820e-04,
            2.12859799e-04,
            1.72638058e-04,
            1.51401327e-04,
            1.47289087e-04,
            1.25616702e-04,
            1.19876706e-04,
            1.06633954e-04
        ])
        np.testing.assert_array_almost_equal(result['detection_scores'][0, :10], expected_detection_scores, decimal=6)

        self.assertTrue(result['detection_classes'].shape == (10, 300))
        expected_detection_classes = np.asarray([81., 71., 66., 15., 63., 71., 66., 84., 64., 37.])
        np.testing.assert_array_almost_equal(result['detection_classes'][0, :10], expected_detection_classes, decimal=6)

        self.assertTrue(result['detection_multiclass_scores'].shape == (10, 300, 91))
        expected_detection_multiclass_scores = np.asarray([
            9.9915493e-01,
            1.5380951e-05,
            3.2381786e-06,
            2.3546692e-05,
            1.0490003e-06,
            2.9198272e-05,
            1.9808563e-06,
            6.0102529e-06,
            8.9344621e-06,
            2.8579292e-05
        ])
        np.testing.assert_array_almost_equal(
            result['detection_multiclass_scores'][0, 2, :10], expected_detection_multiclass_scores, decimal=6
        )

        self.assertTrue(result['detection_anchor_indices'].shape == (10, 300))
        expected_detection_anchor_indices = np.asarray([22., 22., 4., 35., 61., 49., 16., 22., 16., 61.])
        np.testing.assert_array_almost_equal(
            result['detection_anchor_indices'][0, :10], expected_detection_anchor_indices, decimal=6
        )

        self.assertTrue(result['num_detections'].shape == (10,))
        expected_num_detections = np.asarray([300., 300., 300., 300., 300., 300., 300., 300., 300., 300.])
        np.testing.assert_array_almost_equal(result['num_detections'], expected_num_detections, decimal=6)

        self.assertTrue(result['raw_detection_boxes'].shape == (10, 300, 4))
        expected_raw_detection_boxes = np.asarray([0.05784893, 0.05130966, 0.41411403, 0.95867515])
        np.testing.assert_array_almost_equal(
            result['raw_detection_boxes'][0, 2, :], expected_raw_detection_boxes, decimal=6
        )

        self.assertTrue(result['raw_detection_scores'].shape == (10, 300, 91))
        expected_raw_detection_scores = np.asarray([
            9.9981636e-01,
            2.3866653e-06,
            2.2101715e-06,
            1.3920785e-05,
            9.3873712e-07,
            4.0993282e-06,
            3.3591269e-07,
            6.7879691e-06,
            2.8425752e-06,
            9.0685753e-06
        ])
        np.testing.assert_array_almost_equal(
            result['raw_detection_scores'][0, 2, :10], expected_raw_detection_scores, decimal=6
        )

    def test_(self):
        result = self.obj_dec.predict(self.x_test_mnist)




if __name__ == "__main__":
    unittest.main()
