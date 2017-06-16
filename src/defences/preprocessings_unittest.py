import unittest

import numpy as np

from src.defences.preprocessings import label_smoothing

class TestLabelSmoothing(unittest.TestCase):

    def test_default(self):

        M, N = 1000, 20

        y = np.zeros((M, N))
        y[(range(M), np.random.choice(range(N), (M)))] = 1.

        smoothed_y = label_smoothing(y)

        self.assertTrue(np.isclose(np.sum(smoothed_y, axis=1),np.ones(M)).all())
        self.assertTrue((np.max(smoothed_y, axis=1) == np.ones(M)*0.9).all())

    def test_customing(self):

        M, N = 1000, 20

        y = np.zeros((M, N))
        y[(range(M), np.random.choice(range(N), (M)))] = 1.

        smoothed_y = label_smoothing(y, max_value=1/N)

        self.assertTrue(np.isclose(np.sum(smoothed_y, axis=1),np.ones(M)).all())
        self.assertTrue((np.max(smoothed_y, axis=1) == np.ones(M)/N).all())
        self.assertTrue(np.isclose(smoothed_y, np.ones((M, N))/N).all())

if __name__ == '__main__':
    unittest.main()
