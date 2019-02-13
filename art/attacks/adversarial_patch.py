from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks.attack import Attack

logger = logging.getLogger(__name__)


class AdversarialPatch(Attack):
    """
    Implementation of the adversarial patch attack.
    Paper link: https://arxiv.org/abs/1712.09665
    """

    def __init__(self, classifier, expectation=None):
        """
        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param expectation: An expectation over transformations to be applied when computing
                            classifier gradients and predictions.
        :type expectation: :class:`.ExpectationOverTransformations`
        """
        super(AdversarialPatch, self).__init__(classifier=classifier, expectation=expectation)
        kwargs = {#'max_translation': max_translation,
                  # 'num_translations': num_translations,
                  # 'max_rotation': max_rotation,
                  # 'num_rotations': num_rotations
                  }
        self.set_params(**kwargs)

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        logger.info('Computing spatial transformation based on grid search.')

        self.set_params(**kwargs)


        x_adv = None

        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        # :param max_translation: The maximum translation in any direction as percentage of image size.
        # :type max_translation: `float`
        # :param num_translations: The number of translations to search on grid spacing per direction.
        # :type num_translations: `int`
        # :param max_rotation: The maximum rotation in degrees.
        # :type max_rotation: `float`
        # :param num_rotations: The number of rotations to search on grid spacing.
        # :type num_rotations: `int`
        :param expectation: An expectation over transformations to be applied when computing
                            classifier gradients and predictions.
        :type expectation: :class:`.ExpectationOverTransformations`
        """
        super(AdversarialPatch, self).set_params(**kwargs)

        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)

        # if not isinstance(self.max_translation, (float, int)) or self.max_translation < 0 or self.max_translation > 100:
        #     raise ValueError("The maximum translation must be in the range [0, 100].")
        #
        # if not isinstance(self.num_translations, int) or self.num_translations <= 0:
        #     raise ValueError("The number of translations must be a positive integer.")
        #
        # if not isinstance(self.max_rotation, (float, int)) or self.max_rotation < 0 or self.max_translation > 180:
        #     raise ValueError("The maximum rotation must be in the range [0, 180].")
        #
        # if not isinstance(self.num_rotations, int) or self.num_rotations <= 0:
        #     raise ValueError("The number of rotations must be a positive integer.")

        return True
