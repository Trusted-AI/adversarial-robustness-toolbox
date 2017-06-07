import random

import numpy as np
import tensorflow as tf

from src.attackers.attack import Attack, model_loss, clip_perturbation
from src.attackers.deepfool import DeepFool
from src.utils import get_labels_tf_tensor, get_labels_np_array

def universal_perturbation(x_val, model, session, delta=0.2, max_iter=50, eps=10, p=np.inf, max_method_iter=50,
                           clip_min=None, clip_max=None, verbose=1):

    # init universal perturbation
    v = 0
    fooling_rate = 0.0

    dims = list(x_val.shape)
    nb_instances = dims[0]
    dims[0] = None
    xi_op = tf.placeholder(dtype=tf.float32, shape=dims)

    # Compute loss and gradients
    y = get_labels_tf_tensor(model(xi_op))
    loss = model_loss(y, model(xi_op), mean=False)
    grad_xi, = tf.gradients(loss, xi_op)

    attacker = DeepFool(model, session)
    attacker.set_params(clip_min=clip_min, clip_max=clip_max, verbose=0)

    true_y = model.predict(x_val)

    nb_iter = 0
    while (fooling_rate < 1.-delta) and (nb_iter < max_iter):

        rnd_idx = random.sample(range(nb_instances), nb_instances)

        # Go through the data set and compute the perturbation increments sequentially
        for j, x in enumerate(x_val[rnd_idx]):
            xi = x[None, ...]

            f_xi, _ = session.run([model(xi_op), grad_xi], feed_dict={xi_op: xi+v})
            fk_i_hat = np.argmax(f_xi[0])

            fk_hat = np.argmax(true_y[rnd_idx][j])

            if (fk_i_hat == fk_hat):

                # Compute adversarial perturbation
                adv_xi, exe_iter = attacker.generate(xi+v)

                # if it converged, update v
                if exe_iter < max_method_iter:
                    v += adv_xi - xi

                    # Project on l_p ball
                    v = clip_perturbation(v, eps, p)

        nb_iter += 1

        # Compute the error rate
        adv_y = model.predict(x_val + v)
        fooling_rate = np.sum(true_y != adv_y)/nb_instances

    return v, nb_iter, fooling_rate