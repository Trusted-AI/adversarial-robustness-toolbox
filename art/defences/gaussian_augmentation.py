from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class GaussianAugmentation(Preprocessor):
    """
    Add Gaussian noise to a dataset in one of two ways: either add noise to each sample (keeping the size of the
    original dataset) or perform augmentation by keeping all original samples and adding noisy counterparts. When used
    as part of a :class:`.Classifier` instance, the defense will be applied automatically only when training if
    `augmentation` is true, and only when performing prediction otherwise.
    """
    params = ['sigma', 'augmentation', 'ratio']

    def __init__(self, sigma=1., augmentation=True, ratio=1.):
        """
        Initialize a Gaussian augmentation object.

        :param sigma: Standard deviation of Gaussian noise to be added.
        :type sigma: `float`
        :param augmentation: If true, perform dataset augmentation using `ratio`, otherwise replace samples with noisy
                            counterparts.
        :type augmentation: `bool`
        :param ratio: Percentage of data augmentation. E.g. for a rate of 1, the size of the dataset will double.
                      If `augmentation` is false, `ratio` value is ignored.
        :type ratio: `float`
        """
        super(GaussianAugmentation, self).__init__()
        self._is_fitted = True
        self.set_params(sigma=sigma, augmentation=augmentation, ratio=ratio)

    @property
    def apply_fit(self):
        return self.augmentation

    @property
    def apply_predict(self):
        return not self.augmentation

    def __call__(self, x, y=None, sigma=None, augmentation=None, ratio=None):
        """
        Augment the sample `(x, y)` with Gaussian noise. The result is either an extended dataset containing the original
        sample, as well as the newly created noisy samples (augmentation=True) or just the noisy counterparts to the
        original samples.

        :param x: Sample to augment with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels for the sample. If this argument is provided, it will be augmented with the corresponded
                  original labels of each sample point.
        :type y: `np.ndarray`
        :param sigma: Standard deviation of Gaussian noise to be added.
        :type sigma: `float`
        :param augmentation: If true, perform dataset augmentation using `ratio`, otherwise replace samples with noisy
                            counterparts.
        :type augmentation: `bool`
        :param ratio: Percentage of data augmentation. E.g. for a ratio of 1, the size of the dataset will double.
        :type ratio: `float`
        :return: The augmented dataset and (if provided) corresponding labels.
        :rtype:
        """
        # Set params
        params = {}
        if sigma is not None:
            params['sigma'] = sigma

        if augmentation is not None:
            params['augmentation'] = augmentation

        if ratio is not None:
            params['ratio'] = ratio

        if params:
            self.set_params(**params)

        logger.info('Original dataset size: %d', x.shape[0])

        # Select indices to augment
        if self.augmentation:
            size = int(x.shape[0] * self.ratio)
            indices = np.random.randint(0, x.shape[0], size=size)

            # Generate noisy samples
            x_aug = np.random.normal(x[indices], scale=self.sigma, size=(size,) + x[indices].shape[1:])
            x_aug = np.vstack((x, x_aug))
            logger.info('Augmented dataset size: %d', x_aug.shape[0])

            if y is not None:
                return x_aug, np.concatenate((y, y[indices]))
            else:
                return x_aug, y
        else:
            x_aug = np.random.normal(x, scale=self.sigma, size=x.shape)
            logger.info('Created %i samples with Gaussian noise.')

            return x_aug, y

    def estimate_gradient(self, grad):
        return grad

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
        :param augmentation: If true, perform dataset augmentation using `ratio`, otherwise replace samples with noisy
                            counterparts.
        :type augmentation: `bool`
        :param ratio: Percentage of data augmentation. E.g. for a ratio of 1, the size of the dataset will double.
        :type ratio: `float`
        """
        # Save attack-specific parameters
        super(GaussianAugmentation, self).set_params(**kwargs)

        if self.augmentation and self.ratio <= 0:
            raise ValueError("The augmentation ratio must be positive.")

        return True
