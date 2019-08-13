# MIT License
#
# Copyright (C) IBM Corporation 2018
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

from art.attacks.HCLU import HCLU
from art.classifiers.GPy import GPyGaussianProcessClassifier
from art.utils import load_dataset, master_seed


logger = logging.getLogger('testLogger')


class TestHCLU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        # change iris to binary problem, so it is learnable for GPC
        cls.iris = (x_train, y_train[:, 1]), (x_test, y_test[:, 1])

    def setUp(self):
        master_seed(1234)

    def test_GPy(self):
        (X, y), (x_test, y_test) = self.iris
        # set up GPclassifier
        gpkern = GPy.kern.RBF(np.shape(X)[1])
        m = GPy.models.GPClassification(X, y.reshape(-1, 1), kernel=gpkern)
        m.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
        m.optimize(messages=True, optimizer='lbfgs')
        # get ART classifier + clean accuracy
        m_art = GPyGaussianProcessClassifier(m)
        clean_acc = np.mean(np.argmin(m_art.predict(x_test), axis=1) == y_test)
        # get adversarial examples, accuracy, and uncertainty
        attack = HCLU(m_art, conf=0.9, min_val=-0.0, max_val=1.0)
        adv = attack.generate(x_test)
        adv_acc = np.mean(np.argmin(m_art.predict(adv), axis=1) == y_test)
        unc_f = m_art.predict_uncertainty(adv)
        # not all attacks suceed due to the decision surface landscape of GP, some should
        self.assertTrue(clean_acc > adv_acc)

        # now take into account uncertainty
        attack = HCLU(m_art, unc_increase=0.9,
                      conf=0.9, min_val=0.0, max_val=1.0)
        adv = attack.generate(x_test)
        adv_acc = np.mean(np.argmin(m_art.predict(adv), axis=1) == y_test)
        unc_o = m_art.predict_uncertainty(adv)
        # same above
        self.assertTrue(clean_acc > adv_acc)
        # uncertainty should indeed be lower when used as a constraint
        # however, same as above, crafting might fail
        self.assertTrue(np.mean(unc_f > unc_o) > 0.7)


if __name__ == '__main__':
    unittest.main()
