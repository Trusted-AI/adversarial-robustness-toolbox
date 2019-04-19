from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class LabelSmoothing(Preprocessor):
    """
    Computes a vector of smooth labels from a vector of hard ones. The hard labels have to contain ones for the
    correct classes and zeros for all the others. The remaining probability mass between `max_value` and 1 is
    distributed uniformly between the incorrect classes for each instance.
    """
    params = ['max_value']

    def __init__(self, max_value=.9):
        """
        Create an instance of label smoothing.

        :param max_value: Value to affect to correct label
        :type max_value: `float`
        """
        super(LabelSmoothing, self).__init__()
        self._is_fitted = True
        self.set_params(max_value=max_value)

    @property
    def apply_fit(self):
        return True

    @property
    def apply_predict(self):
        return False

    def __call__(self, x, y, max_value=0.9):
        """
        Apply label smoothing.

        :param x: Input data, will not be modified by this method
        :type x: `np.ndarray`
        :param y: Original vector of label probabilities (one-vs-rest)
        :type y: `np.ndarray`
        :param max_value: Value to affect to correct label
        :type max_value: `float`
        :return: Unmodified input data and the vector of smooth probabilities as correct labels
        :rtype: `(np.ndarray, np.ndarray)`
        """
        self.set_params(max_value=max_value)

        min_value = (1 - max_value) / (y.shape[1] - 1)
        assert max_value >= min_value

        smooth_y = y.copy()
        smooth_y[smooth_y == 1.] = max_value
        smooth_y[smooth_y == 0.] = min_value
        return x, smooth_y

    def estimate_gradient(self, x, grad):
        return grad

    def fit(self, x, y=None, **kwargs):
        """No parameters to learn for this method; do nothing."""
        pass

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        Defense-specific parameters:
        :param max_value: Value to affect to correct label
        :type max_value: `float`
        """
        # Save attack-specific parameters
        super(LabelSmoothing, self).set_params(**kwargs)

        if self.max_value <= 0 or self.max_value > 1:
            raise ValueError("The maximum value for correct labels must be between 0 and 1.")

        return True
