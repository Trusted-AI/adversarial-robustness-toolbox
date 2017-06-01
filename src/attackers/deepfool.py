import numpy as np
import tensorflow as tf

from src.attackers.attack import Attack


def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    if mean:
        logits = tf.reduce_mean(logits)
    return logits


class DeepFool(Attack):
    """
    
    """
    def __init__(self, model, sess=None):
        """
        Create a DeepFool attack instance.
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        """
        super(DeepFool, self).__init__(model, sess)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :return: A Numpy array holding the adversarial examples.
        """
        dims = list(x_val.shape)
        dims[0] = None
        xi_op = tf.placeholder(dtype=tf.float32, shape=dims)

        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(self.model(xi_op), 1, keep_dims=True)
        y = tf.to_float(tf.equal(self.model(xi_op), preds_max))
        y = y / tf.reduce_sum(y, 1, keep_dims=True)

        # Compute loss
        loss = model_loss(y, self.model(xi_op), mean=False)
        grad_xi, = tf.gradients(loss, xi_op)

        x_adv = np.empty(x_val.shape)
        for j, x in enumerate(x_val):
            x = x[None, ...]
            f, grd = self.sess.run([self.model(xi_op), grad_xi], feed_dict={xi_op: x})
            fk_hat = np.argmax(f[0])

            xi = x
            fk_i_hat = fk_hat
            f_xi = f[0, fk_hat]
            grd_xi = grd[0, fk_hat]
            while fk_i_hat == fk_hat:
                grad_diff = grd - grd_xi
                f_diff = f - f_xi

                # Masking time
                mask = [0] * f.shape[1]
                mask[fk_hat] = 1
                value = np.ma.array(abs(f_diff) / pow(np.linalg.norm(grad_diff), 2), mask=mask)
                l = value.argmin(fill_value=np.inf)
                r = np.abs(f_diff[0, l]) / pow(np.linalg.norm(grad_diff[0, l]), 2) * grad_diff[0]
                xi += r

                # Recompute prediction for new xi
                f, grd = self.sess.run([self.model(xi_op), grad_xi], feed_dict={xi_op: xi})
                fk_i_hat = np.argmax(f[0])
                grd_xi = grd[0, fk_i_hat]
                f_xi = f[0, fk_i_hat]
            x_adv[j] = xi[0]

        return x_adv
