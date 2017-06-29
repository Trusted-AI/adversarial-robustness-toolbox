import random

import numpy as np
import tensorflow as tf

from keras import backend as K

from src.attackers.attack import Attack, get_logits, clip_perturbation
from src.attackers.deepfool import DeepFool


class UniversalPerturbation(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2016).
    Paper link: https://arxiv.org/abs/1610.08401
    """
    attacks_dict = {"deepfool": DeepFool}
    attack_params = ['attacker', 'attacker_params', 'delta', 'max_iter', 'eps', 'p', 'max_method_iter', 'verbose']

    def __init__(self, model, sess=None, attacker='deepfool', attacker_params=None, delta=0.2, max_iter=5, eps=10,
                 p=np.inf, max_method_iter=50, verbose=1):
        super(UniversalPerturbation, self).__init__(model, sess)
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
            a_instance = self.attacks_dict[a_name](self.model, self.sess)

            if params:
                a_instance.set_params(**params)

            return a_instance

        except KeyError:
            raise NotImplementedError("{} attack not supported".format(a_name))

    def generate(self, x_val, **kwargs):
        self.set_params(**kwargs)

        # init universal perturbation
        v = 0
        fooling_rate = 0.0

        dims = list(x_val.shape)
        nb_instances = dims[0]
        dims[0] = None
        xi_op = tf.placeholder(dtype=tf.float32, shape=dims)

        # Compute loss and gradients
        loss = get_logits(self.model(xi_op), mean=False)

        attacker = self._get_attack(self.attacker, self.attacker_params)

        true_y = self.model.predict(x_val)

        nb_iter = 0
        while (fooling_rate < 1.-self.delta) and (nb_iter < self.max_iter):

            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for j, x in enumerate(x_val[rnd_idx]):
                xi = x[None, ...]

                f_xi = self.sess.run(self.model(xi_op), feed_dict={xi_op: xi+v, K.learning_phase(): 0})
                fk_i_hat = np.argmax(f_xi)

                fk_hat = np.argmax(true_y[rnd_idx][j])

                if fk_i_hat == fk_hat:

                    # Compute adversarial perturbation
                    adv_xi = attacker.generate(np.expand_dims(x+v, 0))

                    adv_f_xi = self.sess.run([self.model(xi_op)], feed_dict={xi_op: adv_xi, K.learning_phase(): 0})
                    adv_fk_i_hat = np.argmax(adv_f_xi[0])

                    # if the class has changed, update v
                    if fk_i_hat != adv_fk_i_hat:
                        v += adv_xi - xi

                        # Project on l_p ball
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
        super(UniversalPerturbation, self).set_params(**kwargs)
