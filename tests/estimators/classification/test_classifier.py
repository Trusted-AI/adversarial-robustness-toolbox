# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

import numpy as np

from art.estimators.classification.classifier import ClassGradientsMixin, ClassifierMixin
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class ClassifierInstance(ClassifierMixin, BaseEstimator):
    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params

    def __init__(self, clip_values=None, channels_first=True):
        super(ClassifierInstance, self).__init__(model=None, clip_values=clip_values)

    def fit(self, x, y, **kwargs):
        pass

    def predict(self, x, **kwargs):
        pass

    def nb_classes(self):
        pass

    def save(self, filename, path=None):
        pass

    def input_shape(self):
        pass


class ClassifierNeuralNetworkInstance(
    ClassGradientsMixin, ClassifierMixin, NeuralNetworkMixin, LossGradientsMixin, BaseEstimator
):
    estimator_params = (
        BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params + ClassifierMixin.estimator_params
    )

    def __init__(self, clip_values, channels_first=True):
        super(ClassifierNeuralNetworkInstance, self).__init__(
            model=None, clip_values=clip_values, channels_first=channels_first
        )

    def class_gradient(self, x, label=None, **kwargs):
        pass

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        pass

    def get_activations(self, x, layer, batch_size):
        pass

    def compute_loss(self, x, y, **kwargs):
        pass

    def loss_gradient(self, x, y, **kwargs):
        pass

    def predict(self, x, batch_size=128, **kwargs):
        pass

    def nb_classes(self):
        pass

    def save(self, filename, path=None):
        pass

    def layer_names(self):
        pass

    def input_shape(self):
        pass


class TestClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

    def setUp(self):
        master_seed(seed=1234)
        super().setUp()

    def test_preprocessing_normalisation(self):
        classifier = ClassifierInstance()

        x = np.random.rand(2, 3)
        x_new, _ = classifier._apply_preprocessing(x=x, y=None, fit=False)
        x_new_expected = np.asarray([[0.19151945, 0.62210877, 0.43772774], [0.78535858, 0.77997581, 0.27259261]])
        np.testing.assert_array_almost_equal(x_new, x_new_expected)

    def test_repr(self):
        classifier = ClassifierInstance()

        repr_ = repr(classifier)
        self.assertIn("ClassifierInstance", repr_)
        self.assertIn("clip_values=None", repr_)
        self.assertIn("defences=None", repr_)
        self.assertIn(
            "preprocessing=StandardisationMeanStd(mean=0.0, std=1.0, apply_fit=True, apply_predict=True)", repr_
        )


class TestClassifierNeuralNetwork(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

    def setUp(self):
        master_seed(seed=1234)
        super().setUp()

    def test_preprocessing_normalisation(self):
        classifier = ClassifierNeuralNetworkInstance((0, 1))
        x = np.random.rand(2, 3)
        x_new_expected = np.asarray([[0.19151945, 0.62210877, 0.43772774], [0.78535858, 0.77997581, 0.27259261]])
        x_new, _ = classifier._apply_preprocessing(x, y=None, fit=False)
        np.testing.assert_array_almost_equal(x_new, x_new_expected, decimal=4)

    def test_repr(self):
        classifier = ClassifierNeuralNetworkInstance((0, 1))
        repr_ = repr(classifier)
        self.assertIn("ClassifierNeuralNetworkInstance", repr_)
        self.assertIn("channels_first=True", repr_)
        self.assertIn("clip_values=[0. 1.]", repr_)
        self.assertIn("defences=None", repr_)
        self.assertIn(
            "preprocessing=StandardisationMeanStd(mean=0.0, std=1.0, apply_fit=True, apply_predict=True)", repr_
        )


if __name__ == "__main__":
    unittest.main()
