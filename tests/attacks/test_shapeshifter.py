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
class TestShapeShifter(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()


if __name__ == "__main__":
    unittest.main()
