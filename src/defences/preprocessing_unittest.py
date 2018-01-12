import unittest

import numpy as np
import tensorflow as tf

from src.defences.preprocessing import feature_squeezing, label_smoothing, tf_feature_squeezing


class TestLabelSmoothing(unittest.TestCase):
    def test_default(self):
        m, n = 1000, 20
        y = np.zeros((m, n))
        y[(range(m), np.random.choice(range(n), m))] = 1.

        smooth_y = label_smoothing(y)
        self.assertTrue(np.isclose(np.sum(smooth_y, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(smooth_y, axis=1) == np.ones(m)*0.9).all())

    def test_customizing(self):
        m, n = 1000, 20
        y = np.zeros((m, n))
        y[(range(m), np.random.choice(range(n), m))] = 1.

        smooth_y = label_smoothing(y, max_value=1./n)
        self.assertTrue(np.isclose(np.sum(smooth_y, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(smooth_y, axis=1) == np.ones(m) / n).all())
        self.assertTrue(np.isclose(smooth_y, np.ones((m, n)) / n).all())


class TestFeatureSqueezing(unittest.TestCase):
    def test_ones(self):
        m, n = 10, 2
        x = np.ones((m, n))

        for depth in range(1,50):
            with self.subTest("bit depth = {}".format(depth)):
                squeezed_x = feature_squeezing(x, depth)
                self.assertTrue((squeezed_x == 1).all())

    def test_random(self):
        m, n = 1000, 20
        x = np.random.rand(m, n)
        x_zero = np.where(x < 0.5)
        x_one = np.where(x >= 0.5)

        squeezed_x = feature_squeezing(x, 1)
        self.assertTrue((squeezed_x[x_zero] == 0.).all())
        self.assertTrue((squeezed_x[x_one] == 1.).all())

        squeezed_x = feature_squeezing(x, 2)
        self.assertFalse(np.logical_and(0. < squeezed_x, squeezed_x < 0.33).any())
        self.assertFalse(np.logical_and(0.34 < squeezed_x, squeezed_x < 0.66).any())
        self.assertFalse(np.logical_and(0.67 < squeezed_x, squeezed_x < 1.).any())

    def test_tf_feature_squeezing(self):
        # With tensors
        m, n = 10, 2
        sess = tf.Session()
        x = tf.ones((m, n))

        for depth in range(1, 10):
            with self.subTest("bit depth = {}".format(depth)):
                squeezed_x = sess.run(tf_feature_squeezing(x, depth))
                self.assertTrue((squeezed_x == 1).all())

        # With placeholders
        x = np.ones((m, n))

        x_op = tf.placeholder(tf.float32, shape=[None, 2])
        for depth in range(1, 10):
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
