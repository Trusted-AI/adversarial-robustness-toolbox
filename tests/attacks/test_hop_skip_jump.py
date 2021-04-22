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
import keras.backend as k
import numpy as np

from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.estimators.classification.keras import KerasClassifier
from art.utils import random_targets

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr
from tests.utils import get_tabular_classifier_pt, master_seed
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestHopSkipJump(TestBase):
    """
    A unittest class for testing the HopSkipJump attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True, set_torch=True)
        super().setUpClass()

        cls.n_train = 100
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def setUp(self):
        master_seed(seed=1234, set_tensorflow=True, set_torch=True)
        super().setUp()

    def test_3_tensorflow_mnist(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf()

        # First targeted attack and norm=2
        hsj = HopSkipJump(classifier=tfc, targeted=True, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # First targeted attack and norm=np.inf
        hsj = HopSkipJump(
            classifier=tfc, targeted=True, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Second untargeted attack and norm=2
        hsj = HopSkipJump(classifier=tfc, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = hsj.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(tfc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Second untargeted attack and norm=np.inf
        hsj = HopSkipJump(
            classifier=tfc, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        x_test_adv = hsj.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(tfc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_8_keras_mnist(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build KerasClassifier
        krc = get_image_classifier_kr()

        # First targeted attack and norm=2
        hsj = HopSkipJump(classifier=krc, targeted=True, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, krc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # First targeted attack and norm=np.inf
        hsj = HopSkipJump(
            classifier=krc, targeted=True, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        params = {"y": random_targets(self.y_test_mnist, krc.nb_classes)}
        x_test_adv = hsj.generate(self.x_test_mnist, **params)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        params.update(mask=mask)
        x_test_adv = hsj.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Second untargeted attack and norm=2
        hsj = HopSkipJump(classifier=krc, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = hsj.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(krc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Second untargeted attack and norm=np.inf
        hsj = HopSkipJump(
            classifier=krc, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        x_test_adv = hsj.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        y_pred = np.argmax(krc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((y_pred != y_pred_adv).any())

        # Test the masking 1
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test the masking 2
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape[1:]))
        mask = mask.reshape(self.x_test_mnist.shape[1:])

        x_test_adv = hsj.generate(self.x_test_mnist, mask=mask)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        # Clean-up session
        k.clear_session()

    # def test_4_pytorch_classifier(self):
    #     """
    #     Third test with the PyTorchClassifier.
    #     :return:
    #     """
    #     x_test = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
    #     x_test_original = x_test.copy()
    #
    #     # Build PyTorchClassifier
    #     ptc = get_image_classifier_pt()
    #
    #     # First targeted attack and norm=2
    #     hsj = HopSkipJump(classifier=ptc, targeted=True, max_iter=20, max_eval=100, init_eval=10, verbose=False)
    #     params = {"y": random_targets(self.y_test_mnist, ptc.nb_classes)}
    #     x_test_adv = hsj.generate(x_test, **params)
    #
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target == y_pred_adv).any())
    #
    #     # Test the masking 1
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
    #     mask = mask.reshape(x_test.shape)
    #
    #     params.update(mask=mask)
    #     x_test_adv = hsj.generate(x_test, **params)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # Test the masking 2
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape[1:]))
    #     mask = mask.reshape(x_test.shape[1:])
    #
    #     params.update(mask=mask)
    #     x_test_adv = hsj.generate(x_test, **params)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # First targeted attack and norm=np.inf
    #     hsj = HopSkipJump(classifier=ptc, targeted=True, max_iter=5, max_eval=100, init_eval=10, norm=np.Inf,
    #     verbose=False)
    #     params = {"y": random_targets(self.y_test_mnist, ptc.nb_classes)}
    #     x_test_adv = hsj.generate(x_test, **params)
    #
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #
    #     target = np.argmax(params["y"], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target == y_pred_adv).any())
    #
    #     # Test the masking 1
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
    #     mask = mask.reshape(x_test.shape)
    #
    #     params.update(mask=mask)
    #     x_test_adv = hsj.generate(x_test, **params)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # Test the masking 2
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape[1:]))
    #     mask = mask.reshape(x_test.shape[1:])
    #
    #     params.update(mask=mask)
    #     x_test_adv = hsj.generate(x_test, **params)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     # self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # Second untargeted attack and norm=2
    #     hsj = HopSkipJump(classifier=ptc, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
    #     x_test_adv = hsj.generate(x_test)
    #
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #
    #     y_pred = np.argmax(ptc.predict(x_test), axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((y_pred != y_pred_adv).any())
    #
    #     # Test the masking 1
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
    #     mask = mask.reshape(x_test.shape)
    #
    #     x_test_adv = hsj.generate(x_test, mask=mask)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # Test the masking 2
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape[1:]))
    #     mask = mask.reshape(x_test.shape[1:])
    #
    #     x_test_adv = hsj.generate(x_test, mask=mask)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # Second untargeted attack and norm=np.inf
    #     hsj = HopSkipJump(classifier=ptc, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf,
    #     verbose=False)
    #     x_test_adv = hsj.generate(x_test)
    #
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #
    #     y_pred = np.argmax(ptc.predict(x_test), axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((y_pred != y_pred_adv).any())
    #
    #     # Test the masking 1
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
    #     mask = mask.reshape(x_test.shape)
    #
    #     x_test_adv = hsj.generate(x_test, mask=mask)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # Test the masking 2
    #     mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape[1:]))
    #     mask = mask.reshape(x_test.shape[1:])
    #
    #     x_test_adv = hsj.generate(x_test, mask=mask)
    #     mask_diff = (1 - mask) * (x_test_adv - x_test)
    #     self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)
    #
    #     unmask_diff = mask * (x_test_adv - x_test)
    #     self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)
    #
    #     # Check that x_test has not been modified by attack and classifier
    #     self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_5_pytorch_resume(self):
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)

        # Build PyTorchClassifier
        ptc = get_image_classifier_pt()

        # HSJ attack
        hsj = HopSkipJump(classifier=ptc, targeted=True, max_iter=10, max_eval=100, init_eval=10, verbose=False)

        params = {"y": self.y_test_mnist[2:3], "x_adv_init": x_test[2:3]}
        x_test_adv1 = hsj.generate(x_test[0:1], **params)
        diff1 = np.linalg.norm(x_test_adv1 - x_test)

        params.update(resume=True, x_adv_init=x_test_adv1)
        x_test_adv2 = hsj.generate(x_test[0:1], **params)
        params.update(x_adv_init=x_test_adv2)
        x_test_adv2 = hsj.generate(x_test[0:1], **params)
        diff2 = np.linalg.norm(x_test_adv2 - x_test)

        self.assertGreater(diff1, diff2)

    def test_7_keras_iris_clipped(self):
        classifier = get_tabular_classifier_kr()

        # Norm=2
        attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

        # Norm=np.inf
        attack = HopSkipJump(
            classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

        # Clean-up session
        k.clear_session()

    def test_7_keras_iris_unbounded(self):
        classifier = get_tabular_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)

        # Norm=2
        attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

        # Norm=np.inf
        attack = HopSkipJump(
            classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

        # Clean-up session
        k.clear_session()

    def test_2_tensorflow_iris(self):
        classifier, sess = get_tabular_classifier_tf()

        # Test untargeted attack and norm=2
        attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

        # Test untargeted attack and norm=np.inf
        attack = HopSkipJump(
            classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

        # Test targeted attack and norm=2
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = HopSkipJump(classifier, targeted=True, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Success rate of targeted HopSkipJump on Iris: %.2f%%", (acc * 100))

        # Test targeted attack and norm=np.inf
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = HopSkipJump(
            classifier, targeted=True, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
        acc = np.sum(preds_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Success rate of targeted HopSkipJump on Iris: %.2f%%", (acc * 100))

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_4_pytorch_iris(self):
        classifier = get_tabular_classifier_pt()
        x_test = self.x_test_iris.astype(np.float32)

        # Norm=2
        attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

        # Norm=np.inf
        attack = HopSkipJump(
            classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
        )
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with HopSkipJump adversarial examples: %.2f%%", (acc * 100))

    def test_6_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        from art.estimators.classification.scikitlearn import SklearnClassifier

        scikitlearn_test_cases = [
            DecisionTreeClassifier(),
            ExtraTreeClassifier(),
            AdaBoostClassifier(),
            BaggingClassifier(),
            ExtraTreesClassifier(n_estimators=10),
            GradientBoostingClassifier(n_estimators=10),
            RandomForestClassifier(n_estimators=10),
            LogisticRegression(solver="lbfgs", multi_class="auto"),
            SVC(gamma="auto"),
            LinearSVC(),
        ]

        x_test_original = self.x_test_iris.copy()

        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)

            # Norm=2
            attack = HopSkipJump(classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, verbose=False)
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info(
                "Accuracy of " + classifier.__class__.__name__ + " on Iris with HopSkipJump adversarial "
                "examples: %.2f%%",
                (acc * 100),
            )

            # Norm=np.inf
            attack = HopSkipJump(
                classifier, targeted=False, max_iter=20, max_eval=100, init_eval=10, norm=np.Inf, verbose=False
            )
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())

            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info(
                "Accuracy of " + classifier.__class__.__name__ + " on Iris with HopSkipJump adversarial "
                "examples: %.2f%%",
                (acc * 100),
            )

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(HopSkipJump, [BaseEstimator, ClassifierMixin])


if __name__ == "__main__":
    unittest.main()
