# MIT License
#
# Copyright (C) IBM Corporation 2019
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


import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

from art.classifiers import KerasClassifier
NUMPY_DTYPE = np.float64

class FunctionallyEquivalentExtraction:
    """

    | Paper link: https://arxiv.org/abs/1909.01838
    """

    def __init__(self, classifier, num_classes, num_neurons):
        self.classifier = classifier
        self.num_classes = num_classes
        self.num_neurons = num_neurons
        self.num_features = np.prod(classifier.input_shape)

        self.u = np.random.normal(0, 1, (1, self.num_features)).astype(dtype=NUMPY_DTYPE)
        self.v = np.random.normal(0, 1, (1, self.num_features)).astype(dtype=NUMPY_DTYPE)

        self.critical_points = list()

    def extract(self):
        self._critical_point_search()
        self._weight_recovery()
        self._sign_recovery()
        self._last_layer_extraction()

    def OL(self, t_i, e_j=None):
        x = self.u + t_i * self.v

        if e_j is not None:
            x = x + e_j

        return self.classifier.predict(x)[0, :].astype(NUMPY_DTYPE)

    def OL_x(self, x):
        return self.classifier.predict(x)[0, :].astype(NUMPY_DTYPE)

    def _critical_point_search(self):

        h_square = self.num_neurons * self.num_neurons / 5
        delta_0 = 0.05

        t = -h_square
        while t < h_square:

            delta = delta_0

            found_critical_point = False

            while not found_critical_point:

                epsilon = delta / 10

                t_1 = t
                t_2 = t + delta

                m_1 = (self.OL(t_1 + epsilon) - self.OL(t_1)) / epsilon
                m_2 = (self.OL(t_2) - self.OL(t_2 - epsilon)) / epsilon

                y_1 = self.OL(t_1)
                y_2 = self.OL(t_2)

                a = t_1
                b = t_2

                if np.sum(np.abs((m_1 - m_2) / m_1) < 0.0001) > 0.3 * self.num_classes:
                    t = t_2
                    break

                x = a + np.divide(y_2 - y_1 - (b - a) * m_2, m_1 - m_2)

                y_hat = y_1 + m_1 * np.divide(y_2 - y_1 - (b - a) * m_2, m_1 - m_2)

                x_mean = np.mean(x[x != -np.inf])

                y = self.OL(x_mean)

                m_x_1 = (self.OL(x_mean + epsilon) - self.OL(x_mean)) / epsilon
                m_x_2 = (self.OL(x_mean) - self.OL(x_mean - epsilon)) / epsilon

                if np.sum(
                        np.abs((y_hat - y) / y) < 0.000001) > 0.3 * self.num_classes and t_1 < x_mean < t_2 and np.sum(
                    np.abs((m_x_1 - m_x_2) / m_x_1) > 0.00001) > 0.3 * self.num_classes:
                    found_critical_point = True
                    self.critical_points.append(x_mean)
                    t = t_2

                else:
                    delta = delta / 2

        print('count:', len(self.critical_points))

    def _weight_recovery(self):

        # Absolute Value Recovery

        d2_OL_d2ej_xi = np.zeros((self.num_features, self.num_neurons), dtype=NUMPY_DTYPE)

        for i in range(self.num_neurons):

            print('i:', i)

            for j in range(self.num_features):

                d = 0.1

                e_j = np.zeros((1, self.num_features))

                d2_OL_d2ej_xi_ok = False

                while not d2_OL_d2ej_xi_ok:

                    e_j[0, j] = d

                    d_OL_dej_xi_p_cej = (self.OL(self.critical_points[i], e_j=e_j) - self.OL(
                        self.critical_points[i])) / d
                    d_OL_dej_xi_m_cej = (self.OL(self.critical_points[i]) - self.OL(self.critical_points[i],
                                                                                    e_j=-e_j)) / d

                    d2_OL_d2ej_xi[j, i] = np.sum(np.abs(d_OL_dej_xi_p_cej - d_OL_dej_xi_m_cej)) / d

                    if d2_OL_d2ej_xi[j, i] < 4e-4 and d < 50:
                        d = d + 0.01
                    else:
                        d2_OL_d2ej_xi_ok = True

        self.A0_pairwise_ratios = np.zeros((self.num_features, self.num_neurons), dtype=NUMPY_DTYPE)

        for i in range(0, self.num_neurons):
            for k in range(0, self.num_features):
                self.A0_pairwise_ratios[k, i] = d2_OL_d2ej_xi[0, i] / d2_OL_d2ej_xi[k, i]

        # Weight Sign Recovery

        for i in range(self.num_neurons):

            print('i:', i)

            for j in range(self.num_features):

                d_0 = 0.02

                e_j = np.zeros((1, self.num_features), dtype=NUMPY_DTYPE)

                e_j[0, 0] += d_0
                e_j[0, j] += d_0

                d_OL_dejek_xi_p_cejek = (self.OL(self.critical_points[i], e_j=e_j) - self.OL(
                    self.critical_points[i])) / d_0
                d_OL_dejek_xi_m_cejek = (self.OL(self.critical_points[i]) - self.OL(self.critical_points[i],
                                                                                    e_j=-e_j)) / d_0

                d2_OL_dejek_xi = (d_OL_dejek_xi_p_cejek - d_OL_dejek_xi_m_cejek)

                if j == 0:
                    d2_OL_dejek_xi_0 = d2_OL_dejek_xi / 2.0

                co_p = np.sum(np.abs(d2_OL_dejek_xi_0 * (1 + 1 / self.A0_pairwise_ratios[j, i]) - d2_OL_dejek_xi))
                co_m = np.sum(np.abs(d2_OL_dejek_xi_0 * (1 - 1 / self.A0_pairwise_ratios[j, i]) - d2_OL_dejek_xi))

                if co_m < co_p * np.max(1 / self.A0_pairwise_ratios[:, i]):
                    self.A0_pairwise_ratios[j, i] *= -1

    def _sign_recovery(self):

        A0_pairwise_ratios = 1.0 / self.A0_pairwise_ratios

        self.B0 = np.zeros((self.num_neurons, 1), dtype=NUMPY_DTYPE)

        for i in range(self.num_neurons):
            x_i = (self.u + self.critical_points[i] * self.v).flatten()
            self.B0[i] = - np.matmul(self.A0_pairwise_ratios[:, i], x_i)

        z_0 = np.random.normal(0, 1, (self.num_features,)).astype(dtype=NUMPY_DTYPE)

        def f_z(z_i):
            return np.squeeze(np.matmul(A0_pairwise_ratios.T, np.expand_dims(z_i, axis=0).T))

        result_z = least_squares(f_z, z_0)

        for i in range(self.num_neurons):

            e_i = np.zeros((self.num_neurons, 1), dtype=NUMPY_DTYPE)
            e_i[i, 0] = 10000

            def f_v(v_i):
                return np.squeeze(np.matmul(-A0_pairwise_ratios.T, np.expand_dims(v_i, axis=0).T) - e_i)

            v_0 = np.random.normal(0, 1, (self.num_features))

            result_v_i = least_squares(f_v, v_0)

            value_p = np.sum(np.abs(self.OL_x(np.expand_dims(result_z.x, axis=0)) - (
                self.OL_x(np.expand_dims(result_z.x + result_v_i.x, axis=0)))))
            value_m = np.sum(np.abs(self.OL_x(np.expand_dims(result_z.x, axis=0)) - (
                self.OL_x(np.expand_dims(result_z.x - result_v_i.x, axis=0)))))

            if value_m < value_p:
                A0_pairwise_ratios[:, i] *= -1

        self.A0_pairwise_ratios = A0_pairwise_ratios

    def _last_layer_extraction(self):

        predictions = np.zeros((self.num_neurons, self.num_classes)).astype(dtype=NUMPY_DTYPE)
        x_h = np.zeros((self.num_neurons, self.num_features)).astype(dtype=NUMPY_DTYPE)

        for i in range(self.num_neurons):
            predictions[i, :] = self.OL(self.critical_points[i])
            x_h[i, :] = self.u + self.critical_points[i] * self.v

        A1_B1_0 = np.random.normal(0, 1, ((self.num_neurons + 1) * self.num_classes)).astype(dtype=NUMPY_DTYPE)

        def f_a1_b1(a1_b1_i):

            layer_1 = np.maximum(np.matmul(self.A0_pairwise_ratios.T, x_h.T) + self.B0, 0.0)

            A1 = a1_b1_i[0:self.num_neurons * self.num_classes].reshape(self.num_neurons, self.num_classes)
            B1 = a1_b1_i[self.num_neurons * self.num_classes:].reshape(self.num_classes, 1)

            layer_2 = np.matmul(A1.T, layer_1) + B1

            return np.squeeze((layer_2.T - predictions).flatten())

        result_a1_b1 = least_squares(f_a1_b1, A1_B1_0)

        A1 = result_a1_b1.x[0:self.num_neurons * self.num_classes].reshape(self.num_neurons, self.num_classes)
        B1 = result_a1_b1.x[self.num_neurons * self.num_classes:].reshape(self.num_classes, 1)

        x_test = x_h

        layer_1 = np.maximum(np.matmul(self.A0_pairwise_ratios.T, x_test.T) + self.B0, 0.0)
        layer_2 = np.matmul(A1.T, layer_1) + B1

        for i in range(self.num_neurons):
            plt.figure()
            plt.plot(layer_2[:, i], '-o')
            plt.plot(self.OL(self.critical_points[i]))

        plt.show()


if __name__ == '__main__':
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    np.random.seed(0)

    if os.path.isfile('./model.h5'):
        model = tf.keras.models.load_model('./model.h5')
    else:
        raise Exception('Model not found.')

    classifier = KerasClassifier(model=model, use_logits=True, clip_values=(0, 1))

    fee = FunctionallyEquivalentExtraction(classifier=classifier, num_classes=10, num_neurons=16)
    fee.extract()
