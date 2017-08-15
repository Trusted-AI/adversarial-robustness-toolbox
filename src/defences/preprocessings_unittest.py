import unittest

import numpy as np
import tensorflow as tf

from src.defences.preprocessings import *

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


class TestFeatureSqueezing(unittest.TestCase):

    def test_ones(self):

        M, N = 10, 2

        x = np.ones((M,N))

        for depth in range(1,50):
            with self.subTest("bit depth = {}".format(depth)):
                squeezed_x = feature_squeezing(x, depth)
                self.assertTrue((squeezed_x == 1).all())


    def test_random(self):

        M, N = 1000, 20

        x = np.random.rand(M, N)

        x_zero = np.where(x < 0.5)
        x_one = np.where(x >= 0.5)

        squeezed_x = feature_squeezing(x, 1)
        self.assertTrue((squeezed_x[x_zero] == 0.).all())
        self.assertTrue((squeezed_x[x_one] == 1.).all())

        squeezed_x = feature_squeezing(x, 2)
        self.assertFalse(np.logical_and(0. < squeezed_x, squeezed_x < 0.33).any())
        self.assertFalse(np.logical_and(0.34 < squeezed_x, squeezed_x < 0.66).any())
        self.assertFalse(np.logical_and(0.67 < squeezed_x, squeezed_x < 1.).any())

    def test_tf_squeezing(self):

        sess = tf.Session()

        M, N = 10, 2

        # with tensors
        x = tf.ones((M, N))

        for depth in range(1,10):
            with self.subTest("bit depth = {}".format(depth)):
                squeezed_x = sess.run(tf_feature_squeezing(x, depth))
                self.assertTrue((squeezed_x == 1).all())

        # with placeholders
        x = np.ones((M, N))

        x_op = tf.placeholder(tf.float32, shape=[None, 2])
        for depth in range(1,10):
            with self.subTest("bit depth = {}".format(depth)):
                squeezed_x = sess.run(tf_feature_squeezing(x_op, depth), feed_dict={x_op: x})
                self.assertTrue((squeezed_x == 1).all())

        grad = tf.gradients(tf.round(x_op), x_op)
        grad_val = sess.run(grad, feed_dict={x_op: x})
        print(grad_val.shape)

        grad = tf.gradients(tf_feature_squeezing(x_op, 1), x_op)
        grad_val = sess.run(grad, feed_dict={x_op: x})
        print(grad_val.shape)

if __name__ == '__main__':
    unittest.main()
