from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from art.classifiers import SklearnLogisticRegression

logger = logging.getLogger('testLogger')


class TestSklearnLogisticRegression(unittest.TestCase):

    def test_dev(self):
        lr = SklearnLogisticRegression()
