from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class GaussianAugmentation(Preprocessor):
    """
    Perform Gaussian augmentation on a dataset.
    """
    params = ['sigma', 'ratio']

    def __init__(self, sigma=1., ratio=1.):
        """
        Initialize a Gaussian augmentation object.

        :param sigma: Standard deviation of Gaussian noise to be added.
        :type sigma: `float`
        :param ratio: Percentage of data augmentation. E.g. for a rate of 1, the size of the dataset will double.
        :type ratio: `float`
        """
        super(GaussianAugmentation, self).__init__()
        self._is_fitted = True
        self.set_params(sigma=sigma, ratio=ratio)

    def __call__(self, x, y=None, sigma=None, ratio=None):
        """
        Augment the sample `(x, y)` with Gaussian noise. The result is an extended dataset containing the original
        sample, as well as the newly created noisy samples.

        :param x: Sample to augment with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels for the sample. If this argument is provided, it will be augmented with the corresponded
                  original labels of each sample point.
        :type y: `np.ndarray`
        :param sigma: Standard deviation of Gaussian noise to be added.
        :type sigma: `float`
        :param ratio: Percentage of data augmentation. E.g. for a ratio of 1, the size of the dataset will double.
        :type ratio: `float`
        :return: The augmented dataset and (if provided) corresponding labels.
        :rtype:
        """
        # Set params
        params = {}
        if sigma is not None:
            params['sigma'] = sigma

        if ratio is not None:
            params['ratio'] = ratio

        if params:
            self.set_params(**params)

        logger.info('Original dataset size: %d', x.shape[0])

        # Select indices to augment
        import numpy as np
        size = int(x.shape[0] * self.ratio)
        indices = np.random.randint(0, x.shape[0], size=size)

        # Generate noisy samples
        x_aug = np.random.normal(x[indices], scale=self.sigma, size=(size,) + x[indices].shape[1:])
        x_aug = np.vstack((x, x_aug))
        logger.info('Augmented dataset size: %d', x_aug.shape[0])

        if y is not None:
            y_aug = np.concatenate((y, y[indices]))
            return x_aug, y_aug
        else:
            return x_aug

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param sigma: Standard deviation of Gaussian noise to be added.
        :type sigma: `float`
        :param ratio: Percentage of data augmentation. E.g. for a ratio of 1, the size of the dataset will double.
        :type ratio: `float`
        """
        # Save attack-specific parameters
        super(GaussianAugmentation, self).set_params(**kwargs)

        if self.ratio <= 0:
            raise ValueError("The augmentation ratio must be positive.")

        return True
