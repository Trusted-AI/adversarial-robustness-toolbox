# MIT License
#
# Copyright (C) IBM Corporation 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the evasion attack `ShadowAttackTensorFlowV2` for TensorFlow v2.

| Paper link: https://arxiv.org/abs/2003.08937
"""
import logging

from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.attack import EvasionAttack

logger = logging.getLogger(__name__)


class ShadowAttackTensorFlowV2(EvasionAttack):
    """
    Implementation of the Shadow Attack for TensorFlow v2.

    | Paper link: https://arxiv.org/abs/2003.08937
    """

    attack_params = EvasionAttack.attack_params + [
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier,
        batch_size=32,
    ):
        """
        Create an instance of the :class:`.ShadowAttack`.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        """
        super().__init__(estimator=classifier)

        if isinstance(self.estimator, TensorFlowV2Classifier):
            self._attack = ShadowAttackTensorFlowV2(
                classifier=classifier,
                batch_size=batch_size,
            )
        elif isinstance(self.estimator, PyTorchClassifier):
            self._attack = ShadowAttackPyTorch(
                classifier=classifier,
                batch_size=batch_size,
            )
        else:
            raise NotImplementedError

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs. `x` is expected to have spatial dimensions.
        :type x: `np.ndarray`
        :param y: An array with the original labels to be predicted.
        :type y: `np.ndarray`
        :return: An array holding the adversarial patch.
        :rtype: `np.ndarray`
        """
        logger.info("Creating adversarial patch.")

        assert y is not None, "Adversarial Patch attack requires target values `y`."

        if len(x.shape) == 2:
            raise ValueError(
                "Feature vectors detected. The adversarial patch can only be applied to data with spatial "
                "dimensions."
            )

        return self._attack.generate(x=x, y=y, **kwargs)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        """
        super().set_params(**kwargs)

        if not isinstance(self._attack.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if not self._attack.batch_size > 0:
            raise ValueError("The batch size must be greater than 0.")

        return True
