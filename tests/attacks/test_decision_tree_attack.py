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

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
import numpy as np

from art.attacks.decision_tree_attack import DecisionTreeAttack
from art.classifiers.scikitklearn import ScikitlearnDecisionTreeClassifier

logger = logging.getLogger('testLogger')


class TestdecisionTreeattack(unittest.TestCase):
    """
    A unittest class for testing the decision tree attack.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        digits = load_digits()
        self.X = digits.data
        self.y = digits.target

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_scikitlearn(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.X, self.y)
        clf_art = ScikitlearnDecisionTreeClassifier(clf)
        attack = DecisionTreeAttack(clf_art)
        adv = attack.generate(self.X[:25])
        #all crafting should succeed
        self.assert(np.sum(clf.predict(adv) == clf.predict(self.X[:25])) == 0)
        targets = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
                            4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9])
        adv = attack.generate(self.X[:25], targets)
        #all targeted crafting should succeed as well
        self.assert(np.sum(cl.predict(adv) == targets) == 25.0)


if __name__ == '__main__':
    unittest.main()
