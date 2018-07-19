from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np

from art.attacks import FastGradientMethod
from art.utils import to_categorical, get_labels_np_array

logging.basicConfig(format='%(levelname)s %(asctime)s %(funcName)s:%(lineno)d %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

class BasicIterativeMethod(FastGradientMethod):
    """
    The Basic Iterative Method is the iterative version of FGM and FGSM. If target labels are not specified, the
    attack aims for the least likely class (the prediction with the lowest score) for each input.
    Paper link: https://arxiv.org/abs/1607.02533
    """
    attack_params = FastGradientMethod.attack_params + ['eps_step', 'max_iter']

    def __init__(self, classifier, norm=np.inf, eps=.3, eps_step=0.1, max_iter=20, targeted=False, random_init=False):
        """
        Create a :class:`BasicIterativeMethod` instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param random_init: Whether to start at the original input or a random point within the epsilon ball
        :type random_init: `bool`
        """
        super(BasicIterativeMethod, self).__init__(classifier, norm=norm, eps=eps, targeted=targeted,random_init=random_init)

        if eps_step > eps:
            raise ValueError('The iteration step `eps_step` has to be smaller than the total attack `eps`.')
        self.eps_step = eps_step

        if max_iter <= 0:
            raise ValueError('The number of iterations `max_iter` has to be a positive integer.')
        self.max_iter = int(max_iter)

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :return:
        """
        from art.utils import projection

        self.set_params(**kwargs)

        # Choose least likely class as target prediction for the attack
        adv_x = x.copy()
        if 'y' not in kwargs or kwargs[str('y')] is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')

            # Use model predictions as correct outputs
            y = self.classifier.predict(x)
        else:
            y = kwargs[str('y')]
        logger.debug("y shape %s", y.shape)
        y = np.argmax(y, axis=1)

        targets = to_categorical(y, nb_classes=self.classifier.nb_classes)
        active_indices = range(len(adv_x))

        for _ in range(self.max_iter):
            # Adversarial crafting
            adv_x[active_indices] = self._compute(adv_x[active_indices], targets[active_indices], self.eps_step, self.random_init)
            noise = projection(adv_x[active_indices] - x[active_indices], self.eps, self.norm)
            adv_x[active_indices] = x[active_indices] + noise
            adv_preds = self.classifier.predict(adv_x[active_indices])

            # Update active indices
            active_indices = np.where(np.argmax(targets[active_indices],axis=1) != np.argmax(adv_preds, axis=1))[0]
            # Stop if no more indices left to explore
            if len(active_indices) == 0:
                break

        return adv_x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        """
        # Save attack-specific parameters
        super(BasicIterativeMethod, self).set_params(**kwargs)

        if self.eps_step > self.eps:
            raise ValueError('The iteration step `eps_step` has to be smaller than the total attack `eps`.')

        if self.max_iter <= 0:
            raise ValueError('The number of iterations `max_iter` has to be a positive integer.')

        return True
