# MIT License
#
# Copyright (C) IBM Corporation 2018
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
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from art.classifiers.utils import check_is_fitted
from art.utils import get_labels_np_array


class AdversarialTrainer:
    """
    Class performing adversarial training based on a model architecture and one or multiple attack methods.
    """
    # TODO Define a data augmentation procedure; for now, all attacks are applied to all data instances.
    def __init__(self, classifier, attacks):
        """
        Create an AdversarialTrainer instance.

        :param classifier: model to train adversarially
        :type classifier: :class:`Classifier`
        :param attacks: attacks to use for data augmentation in adversarial training
        :type attacks: :class:`Attack` or `list(Attack)` or `dict(Attack: dict(string: sequence))`.
            E.g. `{fgsm: {'eps': .1, 'clip_min': 0}, 'deepfool': {...}}`
        """
        # TODO add Sequence support for attack parameters
        self.classifier = classifier
        if not isinstance(attacks, (list, dict)):
            attacks = {attacks: {}}
        elif isinstance(attacks, list):
            attacks = {a: {} for a in attacks}
        self.attacks = attacks

    def fit(self, x, y, **kwargs):
        """
        Train a model adversarially. Each attack specified when creating the AdversarialTrainer is applied to all
        samples in the dataset, and only the successful ones (on the source model) are kept for data augmentation.

        :param x: Training set
        :type x: `np.ndarray`
        :param y: Labels
        :type y: `np.ndarray`
        :param kwargs: Dictionary of parameters to be passed on to the `fit` method of the classifier
        :type kwargs: `dict`
        :return: `None`
        """
        x_augmented = list(x.copy())
        y_augmented = list(y.copy())

        # Generate adversarial samples for each attack
        for i, attack in enumerate(self.attacks):
            # Fit the classifier to be used for the attack
            # TODO Do not refit classifier if already fitted
            attack.classifier.fit(x, y, **kwargs)

            # Predict new labels for the adversarial samples generated
            x_adv = attack.generate(x, **self.attacks[attack])
            y_pred = get_labels_np_array(attack.classifier.predict(x_adv))
            x_adv = x_adv[np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)]
            y_adv = y_pred[np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)]

            # Only add successful attacks to augmented dataset
            x_augmented.extend(list(x_adv))
            y_augmented.extend(list(y_adv))

        # Fit the model with the extended dataset
        self.classifier.fit(np.array(x_augmented), np.array(y_augmented), **kwargs)
        self.x = x_augmented
        self.y = y_augmented

    def predict(self, x, **kwargs):
        """
        Perform prediction using the adversarially trained classifier.

        :param x: Test set
        :type x: `np.ndarray`
        :param kwargs: Other parameters
        :type kwargs: `dict`
        :return: Predictions for test set
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)
