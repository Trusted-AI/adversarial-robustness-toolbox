import logging

import functools
import operator

import numpy as np
from scipy.stats import entropy

from art.classifiers.wrapper import ClassifierWrapper
from art.utils import clip_and_round

logger = logging.getLogger(__name__)

class ZooGradientEstimation(ClassifierWrapper):
    """
    Implementation of ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models.

    https://arxiv.org/abs/1708.03999
    """
    attack_params = ['num_batch', 'sigma', 'round_samples']

    def __init__(self, classifier, n, sigma, round_samples=0):
        """
        :param classifier: An instance of a `Classifier` whose loss_gradient is being approximated
        :type classifier: `Classifier`
        :param n:  The number of samples to draw to approximate the gradient
        :type n: `int`
        :param sigma: Size of the step to take +/-
        :type sigma: `float`
        :param round_samples: The resolution of the input domain to round the data to, e.g., 1.0, or 1/255. Set to 0 to disable.
        :type round_samples: `float`
        """
        super(ZooGradientEstimation, self).__init__(classifier)
        self._predict = self.predict
        self.predict = self.__predict
        s = functools.reduce(operator.mul, self.input_shape)
        self.input_size = s
        if n > s or n <= 0:
            logger.warn("Number of batches larger than input size. Will limit batch size to [1,%d]" % s)
        n = max(1, min(n, s))
        self.set_params(num_batch=n, sigma=sigma, round_samples=round_samples)

    def _generate_samples(self, x, epsilon_map):
        """
        Generate samples around the current image

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param epsilon_map: Samples drawn from search space
        :type epsilon_map: `np.ndarray`
        :return: Two arrays of new input samples to approximate gradient
        :rtype: `list(np.ndarray)`
        """
        minus = clip_and_round(np.repeat(x, self.num_basis, axis=0) - epsilon_map, self.clip_values, self.round_samples)
        plus = clip_and_round(np.repeat(x, self.num_basis, axis=0) + epsilon_map, self.clip_values, self.round_samples)
        return minus, plus

    def loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # TODO Add a bias to the sampling, pass in self.p to choice
        # TODO Implement gradient upsampling (e.g., [32 x 32 x 3] to [64 x 64 x 3])
        epsilon_map = np.zeros((self.num_batch, self.input_size))
        epsilon_map[np.arange(self.num_batch), np.random.choice(self.input_size, self.num_batch, replace=False)] = self.sigma
        epsilon_map.reshape([self.num_basis] + list(self.input_shape))

        grads = []
        for i in range(len(x)):
            minus, plus = self._generate_samples(x[i:i+1], epsilon_map)

            # Vectorized; small tests weren't faster
            # ent_vec = np.vectorize(lambda p: entropy(y[i], p), signature='(n)->()')
            # new_y_minus = ent_vec(self.predict(minus))
            # new_y_plus = ent_vec(self.predict(plus))
            # Vanilla
            new_y_minus = np.array([entropy(y[i], p) for p in self.predict(minus)])
            new_y_plus = np.array([entropy(y[i], p) for p in self.predict(plus)])
            
            query_efficient_grad = 2*np.mean(np.multiply(epsilon_map.reshape(self.num_basis, -1), (new_y_plus - new_y_minus).reshape(self.num_basis, -1) / (2*self.sigma)).reshape([-1] + list(self.input_shape)), axis=0)
            grads.append(query_efficient_grad)
        grads = self._apply_processing_gradient(np.array(grads))
        return grads

    def __predict(self, x, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs. Rounds results first.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        return self._predict(clip_and_round(x, self.clip_values, self.round_samples), logits, batch_size)