import random

import numpy as np
import tensorflow as tf

from src.attackers.attack import Attack, model_loss, clip_perturbation
from src.attackers.deepfool import DeepFool
from src.utils import get_labels_tf_tensor, get_labels_np_array


class UniversalPerturbation(Attack):

    attacks_dict = {"deepfool": DeepFool}

    def __init__(self, model, sess=None):
        super(UniversalPerturbation, self).__init__(model, sess)

    def _get_attack(self, a_name, params=None):
        try:
            a_instance = self.attacks_dict[a_name](self.model, self.sess)

            if params:
                a_instance.set_params(**params)

            return a_instance

        except KeyError:
            raise NotImplementedError("{} attack not supported".format(a_name))

    def generate(self, x_val, attacker, attacker_params=None, delta=0.2, max_iter=50, eps=10,
                           p=np.inf, max_method_iter=50, verbose=1):

        # init universal perturbation
        v = 0
        fooling_rate = 0.0

        dims = list(x_val.shape)
        nb_instances = dims[0]
        dims[0] = None
        xi_op = tf.placeholder(dtype=tf.float32, shape=dims)

        # Compute loss and gradients
        y = get_labels_tf_tensor(self.model(xi_op))
        loss = model_loss(y, self.model(xi_op), mean=False)
        grad_xi, = tf.gradients(loss, xi_op)

        attacker = self._get_attack(attacker, attacker_params)

        true_y = self.model.predict(x_val)

        nb_iter = 0
        while (fooling_rate < 1.-delta) and (nb_iter < max_iter):

            rnd_idx = random.sample(range(nb_instances), nb_instances)

            # Go through the data set and compute the perturbation increments sequentially
            for j, x in enumerate(x_val[rnd_idx]):
                xi = x[None, ...]

                f_xi, _ = self.sess.run([self.model(xi_op), grad_xi], feed_dict={xi_op: xi+v})
                fk_i_hat = np.argmax(f_xi[0])

                fk_hat = np.argmax(true_y[rnd_idx][j])

                if (fk_i_hat == fk_hat):

                    # Compute adversarial perturbation
                    adv_xi = attacker.generate(xi+v)

                    adv_f_xi, _ = self.sess.run([self.model(xi_op), grad_xi], feed_dict={xi_op: adv_xi})
                    adv_fk_i_hat = np.argmax(adv_f_xi[0])

                    # if the class has changed, update v
                    if (fk_i_hat != adv_fk_i_hat):
                        v += adv_xi - xi

                        # Project on l_p ball
                        v = clip_perturbation(v, eps, p)

            nb_iter += 1

            # Compute the error rate
            adv_x = x_val + v
            adv_y = self.model.predict(adv_x)
            fooling_rate = np.sum(true_y != adv_y)/nb_instances

        self.fooling_rate = fooling_rate
        self.converged = (nb_iter < max_iter)
        self.v = v

        return adv_x