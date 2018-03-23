from __future__ import absolute_import, division, print_function, unicode_literals

from keras import backend as k
import numpy as np
import tensorflow as tf

from src.attacks.attack import Attack


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). This implementation extends the attack to other norms, and is therefore called the Fast
    Gradient Method. Paper link: https://arxiv.org/abs/1412.6572
    """
    attack_params = ['ord', 'y', 'y_val', 'targeted', 'clip_min', 'clip_max']

    def __init__(self, classifier, sess=None, ord=np.inf, y=None, targeted=False, clip_min=None, clip_max=None):
        """Create a FastGradientMethod instance.
        :param ord: (optional) Order of the norm. Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param targeted: (optional boolean) Should the attack target one specific class
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        super(FastGradientMethod, self).__init__(classifier, sess)

        kwargs = {'ord': ord, 'targeted': targeted, 'clip_min': clip_min, 'clip_max': clip_max, 'y': y}
        self.set_params(**kwargs)

    def generate_graph(self, x, eps, **kwargs):
        """Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional tf.placeholder) The placeholder for input variation (noise amplitude)
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        self.set_params(**kwargs)

        preds = self.classifier._get_predictions(x, log=False)

        if not hasattr(self, 'y') or self.y is None:
            # Use model predictions as correct outputs
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            y = tf.to_float(tf.equal(preds, preds_max))
            y = tf.stop_gradient(y)
        else:
            y = self.y
        y = y / tf.reduce_sum(y, 1, keep_dims=True)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, x)

        # Apply norm bound
        if self.ord == np.inf:
            grad = tf.sign(grad)
        elif self.ord == 1:
            ind = list(range(1, len(x.get_shape())))
            grad = grad / tf.reduce_sum(tf.abs(grad), reduction_indices=ind, keep_dims=True)
        elif self.ord == 2:
            ind = list(range(1, len(x.get_shape())))
            grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=ind, keep_dims=True))

        # Apply perturbation and clip
        x_adv_op = x + eps * grad
        if self.clip_min is not None and self.clip_max is not None:
            x_adv_op = tf.clip_by_value(x_adv_op, self.clip_min, self.clip_max)

        return x_adv_op

    def minimal_perturbations(self, x, x_val, eps_step=0.1, eps_max=1., **kwargs):
        """Iteratively compute the minimal perturbation necessary to make the class prediction change. Stop when the
        first adversarial example was found.

        :param x: (required) A placeholder for the input.
        :param x_val: (required) A Numpy array with the original inputs.
        :param eps_step: (optional float) The increase in the perturbation for each iteration
        :param eps_max: (optional float) The maximum accepted perturbation
        :param kwargs: Other parameters to send to generate_graph
        :return: A Numpy array holding the adversarial examples.
        """
        k.set_learning_phase(0)
        y = np.argmax(self.model.predict(x_val), 1)
        adv_x = x_val.copy()

        curr_indexes = np.arange(len(adv_x))
        eps = tf.placeholder(tf.float32, None)
        adv_x_op = self.generate_graph(x, eps, **kwargs)
        adv_y = tf.argmax(self.model(adv_x_op), 1)
        eps_val = eps_step

        while len(curr_indexes) != 0 and eps_val <= eps_max:
            # Adversarial crafting
            new_adv_x, new_y = self.sess.run([adv_x_op, adv_y], {x: x_val[curr_indexes], eps: eps_val})

            # Update
            adv_x[curr_indexes] = new_adv_x
            curr_indexes = np.where(y[curr_indexes] == new_y)[0]
            eps_val += eps_step

        return adv_x

    def generate(self, x_val, **kwargs):
        """Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :param eps: (required float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :return: A Numpy array holding the adversarial examples.
        """

        input_shape = [None] + list(x_val.shape[1:])
        self._x = tf.placeholder(tf.float32, shape=input_shape)
        k.set_learning_phase(0)

        # Return adversarial examples computed with minimal perturbation if option is active
        if "minimal" in kwargs and kwargs["minimal"]:
            return self.minimal_perturbations(self._x, x_val, **kwargs)

        # Generate computation graph
        self._x_adv = self.generate_graph(self._x, **kwargs)

        # Run symbolic graph without or with true labels
        if 'y_val' not in kwargs or kwargs['y_val'] is None:
            feed_dict = {self._x: x_val}
        else:
            # Verify label placeholder was given in params if using true labels
            if self.y is None:
                raise Exception("True labels given but label placeholder not given.")
            feed_dict = {self._x: x_val, self.y: kwargs['y_val']}

        return self.sess.run(self._x_adv, feed_dict=feed_dict)

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        super(FastGradientMethod, self).set_params(**kwargs)

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True
