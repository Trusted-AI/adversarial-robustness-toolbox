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
from __future__ import absolute_import, division, print_function

import keras.backend as k
import numpy as np
import random
import tensorflow as tf

from art.attacks.attack import Attack, clip_perturbation
from art.attacks.deepfool import DeepFool

# TODO Import other attacks


class UniversalPerturbation(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2016). Computes a fixed perturbation to be applied to all
    future inputs. To this end, it can use any adversarial attack method. DeepFool is the base attack here, as in the
    original paper. Paper link: https://arxiv.org/abs/1610.08401
    """
    attacks_dict = {"deepfool": DeepFool}
    attack_params = ['attacker', 'attacker_params', 'delta', 'max_iter', 'eps', 'p', 'max_method_iter', 'verbose']

    def __init__(self, classifier, sess=None, attacker='deepfool', attacker_params=None, delta=0.2, max_iter=5, eps=10,
                 p=np.inf, max_method_iter=50, verbose=1):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param sess: The session to run graphs in.
        :type sess: `tf.Session`
        :param attacker: Adversarial attack name. Default is 'deepfool'.
        :type attacker: `str`
        :param attacker_params: Parameters specific to the adversarial attack.
        :type attacker_params: `dict`
        :param delta: (float, default 0.2)
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :type max_iter: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param p: Order of the norm. Possible values: np.inf, 1 or 2 (default is np.inf)
        :type p: `int`
        :param max_method_iter: The maximum number of iterations for the attack method (if it applies)
        :type max_method_iter: `int`
        :param verbose: For status updates in progress bar.
        :type verbose: `bool`
        """
        super(UniversalPerturbation, self).__init__(classifier, sess)
        kwargs = {'attacker': attacker,
                  'attacker_params': attacker_params,
                  'delta': delta,
                  'max_iter': max_iter,
                  'eps': eps,
                  'p': p,
                  'max_method_iter': max_method_iter,
                  'verbose': verbose}
        self.set_params(**kwargs)

    def _get_attack(self, a_name, params=None):
        try:
            a_instance = self.attacks_dict[a_name](self.classifier, self.sess)

            if params:
                a_instance.set_params(**params)

            return a_instance

        except KeyError:
            raise NotImplementedError("{} attack not supported".format(a_name))

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x_val: An array with the original inputs.
        :type x_val: `np.ndarray`
        :param attacker: Adversarial attack name. Default is 'deepfool'.
        :type attacker: `str`
        :param attacker_params: Parameters specific to the adversarial attack.
        :type attacker_params: `dict`
        :param delta: (float, default 0.2)
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :type max_iter: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param p: Order of the norm. Possible values: np.inf, 1 or 2 (default is np.inf)
        :type p: `int`
        :param max_method_iter: The maximum number of iterations for the attack method (if it applies)
        :type max_method_iter: `int`
        :param verbose: For status updates in progress bar.
        :type verbose: `bool`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        self.set_params(**kwargs)
        k.set_learning_phase(0)

        # Init universal perturbation
        v = 0
        fooling_rate = 0.0

        dims = list(x_val.shape)
        nb_instances = dims[0]
        dims[0] = None
        xi_op = tf.placeholder(dtype=tf.float32, shape=dims)

        attacker = self._get_attack(self.attacker, self.attacker_params)
        true_y = self.model.predict(x_val)

        nb_iter = 0
        while fooling_rate < 1. - self.delta and nb_iter < self.max_iter:

            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for j, x in enumerate(x_val[rnd_idx]):
                xi = x[None, ...]

                f_xi = self.sess.run(self.model(xi_op), feed_dict={xi_op: xi + v})
                fk_i_hat = np.argmax(f_xi[0])
                fk_hat = np.argmax(true_y[rnd_idx][j])

                if fk_i_hat == fk_hat:

                    # Compute adversarial perturbation
                    adv_xi = attacker.generate(np.expand_dims(x, 0) + v)
                    adv_f_xi = self.sess.run(self.model(xi_op), feed_dict={xi_op: adv_xi})
                    adv_fk_i_hat = np.argmax(adv_f_xi[0])

                    # If the class has changed, update v
                    if fk_i_hat != adv_fk_i_hat:
                        v += adv_xi - xi

                        # Project on L_p ball
                        v = clip_perturbation(v, self.eps, self.p)
            nb_iter += 1

            # Compute the error rate
            adv_x = x_val + v
            adv_y = self.model.predict(adv_x)
            fooling_rate = np.sum(true_y != adv_y)/nb_instances

        self.fooling_rate = fooling_rate
        self.converged = (nb_iter < self.max_iter)
        self.v = v

        return adv_x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param attacker: Adversarial attack name. Default is 'deepfool'.
        :type attacker: `str`
        :param attacker_params: Parameters specific to the adversarial attack.
        :type attacker_params: `dict`
        :param delta: (float, default 0.2)
        :param max_iter: The maximum number of iterations for computing universal perturbation.
        :type max_iter: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param p: Order of the norm. Possible values: np.inf, 1 or 2 (default is np.inf)
        :type p: `int`
        :param max_method_iter: The maximum number of iterations for the attack method (if it applies)
        :type max_method_iter: `int`
        :param verbose: For status updates in progress bar.
        :type verbose: `bool`
        """
        super(UniversalPerturbation, self).set_params(**kwargs)
