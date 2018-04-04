from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

# TODO keep clip function? move it somewhere else?


# def clip_perturbation(v, eps, p):
#     """
#     Clip the values in v if their L_p norm is larger than eps.
#     :param v: array of perturbations to clip
#     :param eps: maximum norm allowed
#     :param p: L_p norm to use for clipping. Only p = 2 and p = Inf supported for now
#     :return: clipped values of v
#     """
#     if p == 2:
#         v *= min(1., eps/np.linalg.norm(v, axis=(1, 2)))
#     elif p == np.inf:
#         v = np.sign(v) * np.minimum(abs(v), eps)
#     else:
#         raise NotImplementedError('Values of p different from 2 and Inf are currently not supported.')
#
#     return v

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Attack(ABC):
    """
    Abstract base class for all attack classes.
    """
    attack_params = ['classifier']

    def __init__(self, classifier):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        """
        self.classifier = classifier

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param x_val: An array with the original inputs to be attacked.
        :type x_val: `np.ndarray`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True
