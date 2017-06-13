from keras.utils.generic_utils import Progbar
import numpy as np
import tensorflow as tf

from src.attackers.attack import Attack, get_logits


class DeepFool(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2015).
    Paper link: https://arxiv.org/abs/1511.04599
    """
    attack_params = ['max_iter', 'clip_min', 'clip_max', 'verbose']

    def __init__(self, model, sess=None, max_iter=50, clip_min=None, clip_max=None, verbose=1):
        """
        Create a DeepFool attack instance.
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        """
        super(DeepFool, self).__init__(model, sess)
        params = {'max_iter': max_iter, 'clip_min': clip_min, 'clip_max': clip_max, 'verbose': verbose}
        self.set_params(**params)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :return: A Numpy array holding the adversarial examples.
        """
        assert self.set_params(**kwargs)

        dims = list(x_val.shape)
        dims[0] = None
        xi_op = tf.placeholder(dtype=tf.float32, shape=dims)
        loss = get_logits(self.model(xi_op), mean=False)
        grad_xi, = tf.gradients(loss, xi_op)
        x_adv = x_val.copy()

        # Progress bar
        progress_bar = Progbar(target=len(x_val), verbose=self.verbose)

        for j, x in enumerate(x_adv):
            xi = x[None, ...]

            f, grd = self.sess.run([self.model(xi_op), grad_xi], feed_dict={xi_op: xi})
            fk_hat = np.argmax(f[0])

            fk_i_hat = fk_hat
            f_xi = f[0, fk_hat]
            grd_xi = grd[0, fk_hat]

            nb_iter = 0

            while (fk_i_hat == fk_hat) and (nb_iter < self.max_iter):

                grad_diff = grd - grd_xi
                f_diff = f - f_xi

                # Masking time
                mask = [0] * f.shape[1]
                mask[fk_hat] = 1
                value = np.ma.array(abs(f_diff) / pow(np.linalg.norm(grad_diff), 2), mask=mask)

                l = value.argmin(fill_value=np.inf)
                r = np.abs(f_diff[0, l]) / pow(np.linalg.norm(grad_diff[0, l]), 2) * grad_diff[0]

                # Add perturbation and clip result
                xi += r

                if self.clip_min or self.clip_max:
                    np.clip(xi, self.clip_min, self.clip_max, xi)

                # Recompute prediction for new xi
                f, grd = self.sess.run([self.model(xi_op), grad_xi], feed_dict={xi_op: xi})
                fk_i_hat = np.argmax(f[0])
                grd_xi = grd[0, fk_i_hat]
                f_xi = f[0, fk_i_hat]

                nb_iter += 1

            progress_bar.update(current=j, values=[("perturbation", abs(np.average(r)))])

        return x_adv

    def set_params(self, **kwargs):
        super(DeepFool, self).set_params(**kwargs)

        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        return True
