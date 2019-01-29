import logging

import numpy as np
from scipy.stats import entropy

from art.classifiers.mixin import ClassifierMixin

logger = logging.getLogger(__name__)

class BlackBoxAttack(object):
    """
    Convenient Mixin class with convenience methods to clip and round values
    to the correct level of granularity to ensure the blackbox attack is valid
    """
    def _clip_and_round(self, x):
        """
        Rounds the input to the correct level of granularity.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        """
        if self.round_samples == 0:
            return x
        x = np.clip(x, *self.clip_values)
        x = np.around(x / self.round_samples) * self.round_samples
        return x

class QueryEfficientBBAttack(ClassifierMixin, BlackBoxAttack):
    """
    Implementation of Query-Efficient Black-box Adversarial Examples
    The attack approximates the gradient by maximizing the loss function
    over samples drawn from random Gaussian noise around the input.

    https://arxiv.org/abs/1712.07113
    """
    
    def __init__(self, classifier, n, sigma, round_samples=0):
        """
        :param classifier: An instance of a `Classifier` whose loss_gradient is being approximated
        :type classifier: `Classifier`
        :param n:  The number of samples to draw to approximate the gradient
        :type n: `int`
        :param sigma: Scaling on the Gaussian noise N(0,1)
        :type sigma: `float`
        :param round_samples: [Not implemented] Whether or not transform samples into the appropriate domain e.g., [0, 255]_Z versus [0,1]
        :type round_samples: `bool`
        """
        super(QueryEfficientBBAttack, self).__init__(classifier)
        object.__setattr__(self, 'num_basis', n)
        object.__setattr__(self, 'sigma', sigma)
        object.__setattr__(self, 'round_samples', round_samples)

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
        minus = self._clip_and_round(np.repeat(x, self.num_basis, axis=0) - epsilon_map)
        plus = self._clip_and_round(np.repeat(x, self.num_basis, axis=0) + epsilon_map)
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
        epsilon_map = self.sigma*np.random.normal(size=([self.num_basis] + list(self.input_shape)))
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