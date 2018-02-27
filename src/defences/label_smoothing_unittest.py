from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np

from src.defences.label_smoothing import LabelSmoothing


class TestLabelSmoothing(unittest.TestCase):
    def test_default(self):
        m, n = 1000, 20
        y = np.zeros((m, n))
        y[(range(m), np.random.choice(range(n), m))] = 1.

        ls = LabelSmoothing()
        _, smooth_y = ls(None, y)
        self.assertTrue(np.isclose(np.sum(smooth_y, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(smooth_y, axis=1) == np.ones(m)*0.9).all())

    def test_customizing(self):
        m, n = 1000, 20
        y = np.zeros((m, n))
        y[(range(m), np.random.choice(range(n), m))] = 1.

        ls = LabelSmoothing()
        _, smooth_y = ls(None, y, max_value=1./n)
        self.assertTrue(np.isclose(np.sum(smooth_y, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(smooth_y, axis=1) == np.ones(m) / n).all())
        self.assertTrue(np.isclose(smooth_y, np.ones((m, n)) / n).all())


if __name__ == '__main__':
    unittest.main()
