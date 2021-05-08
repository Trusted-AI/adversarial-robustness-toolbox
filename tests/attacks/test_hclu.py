# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
import GPy

from art.attacks.evasion.hclu import HighConfidenceLowUncertainty
from art.estimators.classification.GPy import GPyGaussianProcessClassifier

from tests.utils import TestBase
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestHCLU(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # change iris to a binary problem
        cls.x_train = cls.x_train_iris[0:35]
        cls.y_train = cls.y_train_iris[0:35, 1]
        cls.x_test = cls.x_train
        cls.y_test = cls.y_train

    def test_GPy(self):
        x_test_original = self.x_test.copy()
        # set up GPclassifier
        gpkern = GPy.kern.RBF(np.shape(self.x_train)[1])
        m = GPy.models.GPClassification(self.x_train, self.y_train.reshape(-1, 1), kernel=gpkern)
        m.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
        m.optimize(messages=True, optimizer="lbfgs")
        # get ART classifier + clean accuracy
        m_art = GPyGaussianProcessClassifier(m)
        clean_acc = np.mean(np.argmin(m_art.predict(self.x_test), axis=1) == self.y_test)
        # get adversarial examples, accuracy, and uncertainty
        attack = HighConfidenceLowUncertainty(m_art, conf=0.9, min_val=-0.0, max_val=1.0, verbose=False)
        adv = attack.generate(self.x_test)
        adv_acc = np.mean(np.argmin(m_art.predict(adv), axis=1) == self.y_test)
        unc_f = m_art.predict_uncertainty(adv)
        # not all attacks succeed due to the decision surface landscape of GP, some should
        self.assertGreater(clean_acc, adv_acc)

        # now take into account uncertainty
        attack = HighConfidenceLowUncertainty(
            m_art, unc_increase=0.9, conf=0.9, min_val=0.0, max_val=1.0, verbose=False
        )
        adv = attack.generate(self.x_test)
        adv_acc = np.mean(np.argmin(m_art.predict(adv), axis=1) == self.y_test)
        unc_o = m_art.predict_uncertainty(adv)
        # same above
        self.assertGreater(clean_acc, adv_acc)
        # uncertainty should indeed be lower when used as a constraint
        # however, same as above, crafting might fail
        self.assertGreater(np.mean(unc_f > unc_o), 0.65)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(HighConfidenceLowUncertainty, [GPyGaussianProcessClassifier])


if __name__ == "__main__":
    unittest.main()
