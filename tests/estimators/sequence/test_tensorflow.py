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

import pytest
import tensorflow.compat.v1 as tf1
from lingvo.core.hyperparams import Params

from art.estimators.sequence.sequence import SequenceNetworkMixin
from art.estimators.sequence.tensorflow import LingvoAsr
from art.estimators.tensorflow import TensorFlowV2Estimator

logger = logging.getLogger(__name__)


class TestLingvoAsr:
    """
    Test the LingvoAsr estimator.
    """

    def test_is_subclass(self):
        assert issubclass(LingvoAsr, (SequenceNetworkMixin, TensorFlowV2Estimator))

    def test_implements_abstract_methods(self):
        tf1.reset_default_graph()
        LingvoAsr()

    def test_check_and_download_params(self):
        tf1.reset_default_graph()

        lingvo = LingvoAsr()
        assert lingvo._check_and_download_params()

    def test_check_and_download_model(self):
        tf1.reset_default_graph()

        lingvo = LingvoAsr()
        assert lingvo._check_and_download_model()

    def test_check_and_download_vocab(self):
        tf1.reset_default_graph()

        lingvo = LingvoAsr()
        assert lingvo._check_and_download_vocab()

    def test_load_model(self):
        tf1.reset_default_graph()

        LingvoAsr()
        graph = tf1.get_default_graph()
        assert graph.get_operations()
