from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from art.attacks import BasicIterativeMethod
from art.utils import to_categorical, get_labels_np_array


class ProjectedGradientDescent(BasicIterativeMethod):
    """
    The Projected Gradient Descent attack is a variant of the Basic Iterative Method in which,
    after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted
    data range). 
    This is the attack proposed by Madry et al. for adversarial training.
    Paper link: https://arxiv.org/abs/1706.06083
    """
    attack_params = BasicIterativeMethod.attack_params + ['eps_step', 'max_iter']

    def __init__(self, classifier, norm=np.inf, eps=.3, eps_step=0.1, max_iter=20, targeted=False, random_init=False):
        """
        Create a :class:`ProjectedGradientDescent` instance.

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
        super(ProjectedGradientDescent, self).__init__(classifier, norm=norm, eps=eps, targeted=targeted,
                                                   random_init=random_init)

        self._project = True

    