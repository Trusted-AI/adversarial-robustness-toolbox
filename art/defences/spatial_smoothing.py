from __future__ import absolute_import, division, print_function, unicode_literals

from scipy import ndimage

from art.defences.preprocessor import Preprocessor


class SpatialSmoothing(Preprocessor):
    """
    Implement the local spatial smoothing defence approach. Defence method from https://arxiv.org/abs/1704.01155.
    """
    params = ["window_size"]

    def __init__(self, window_size=3):
        """
        Create an instance of local spatial smoothing.

        :param window_size: The size of the sliding window.
        :type window_size: `int`
        """
        super(SpatialSmoothing, self).__init__()
        self._is_fitted = True
        self.set_params(window_size=window_size)

    def __call__(self, x, y=None, window_size=None):
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param window_size: The size of the sliding window.
        :type window_size: `int`
        :return: Smoothed sample
        :rtype: `np.ndarray`
        """
        if window_size is not None:
            self.set_params(window_size=window_size)

        size = (1, self.window_size, self.window_size, 1)
        result = ndimage.filters.median_filter(x, size=size, mode="reflect")

        return result

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param window_size: The size of the sliding window.
        :type window_size: `int`
        """
        # Save attack-specific parameters
        super(SpatialSmoothing, self).set_params(**kwargs)

        if type(self.window_size) is not int or self.window_size <= 0:
            raise ValueError("Sliding window size must be a positive integer.")

        return True
