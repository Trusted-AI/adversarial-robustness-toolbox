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

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """
    Class performing adversarial training based on a model architecture and one or multiple attack methods.

    Incorporates original adversarial training, ensemble adversarial training (https://arxiv.org/abs/1705.07204),
    training on all adversarial data and other common setups. If multiple attacks are specified, they are rotated
    for each batch. If the specified attacks have as target a different model, then the attack is transferred. The
    `ratio` determines how many of the clean samples in each batch are replaced with their adversarial counterpart.

     .. warning:: Both successful and unsuccessful adversarial samples are used for training. In the case of
                  unbounded attacks (e.g., DeepFool), this can result in invalid (very noisy) samples being included.
    """
    def __init__(self, classifier, attacks, ratio=.5):
        """
        Create an :class:`AdversarialTrainer` instance.

        :param classifier: Model to train adversarially.
        :type classifier: :class:`Classifier`
        :param attacks: attacks to use for data augmentation in adversarial training
        :type attacks: :class:`Attack` or `list(Attack)`
        :param ratio: The proportion of samples in each batch to be replaced with their adversarial counterparts.
                      Setting this value to 1 allows to train only on adversarial samples.
        :type ratio: `float`
        """
        from art.attacks import Attack

        self.classifier = classifier
        if isinstance(attacks, Attack):
            self.attacks = [attacks]
        elif isinstance(attacks, list):
            self.attacks = attacks
        else:
            raise ValueError('Only Attack instances or list of attacks supported.')

        if ratio <= 0 or ratio > 1:
            raise ValueError('The `ratio` of adversarial samples in each batch has to be between 0 and 1.')
        self.ratio = ratio

        self._precomputed_adv_samples = []
        self.x_augmented, self.y_augmented = None, None

    def fit(self, x, y, batch_size=128, nb_epochs=20):
        """
        Train a model adversarially. See class documentation for more information on the exact procedure.

        :param x: Training set.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        logger.info('Performing adversarial training using %i attacks.', len(self.attacks))
        nb_batches = int(np.ceil(len(x) / batch_size))
        nb_adv = int(np.ceil(self.ratio * batch_size))
        ind = np.arange(len(x))
        attack_id = 0

        # Precompute adversarial samples for transferred attacks
        logged = False
        self._precomputed_adv_samples = []
        for attack in self.attacks:
            if 'targeted' in attack.attack_params:
                if attack.targeted:
                    raise NotImplementedError("Adversarial training with targeted attacks is \
                                               currently not implemented")

            if attack.classifier != self.classifier:
                if not logged:
                    logger.info('Precomputing transferred adversarial samples.')
                    logged = True
                self._precomputed_adv_samples.append(attack.generate(x, y=y))
            else:
                self._precomputed_adv_samples.append(None)

        for e in range(nb_epochs):
            logger.info('Adversarial training epoch %i/%i', e, nb_epochs)

            # Shuffle the examples
            np.random.shuffle(ind)

            for batch_id in range(nb_batches):
                # Create batch data
                x_batch = x[ind[batch_id * batch_size:min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                y_batch = y[ind[batch_id * batch_size:min((batch_id + 1) * batch_size, x.shape[0])]]

                # Choose indices to replace with adversarial samples
                attack = self.attacks[attack_id]
                adv_ids = np.random.choice(x_batch.shape[0], size=nb_adv, replace=False)

                # If source and target models are the same, craft fresh adversarial samples
                if attack.classifier == self.classifier:
                    x_batch[adv_ids] = attack.generate(x_batch[adv_ids], y=y_batch[adv_ids])

                # Otherwise, use precomputed adversarial samples
                else:
                    x_adv = self._precomputed_adv_samples[attack_id]
                    x_adv = x_adv[ind[batch_id * batch_size:min((batch_id + 1) * batch_size, x.shape[0])]][adv_ids]
                    x_batch[adv_ids] = x_adv

                # Fit batch
                self.classifier.fit(x_batch, y_batch, nb_epochs=1, batch_size=x_batch.shape[0])
                attack_id = (attack_id + 1) % len(self.attacks)

    def predict(self, x, **kwargs):
        """
        Perform prediction using the adversarially trained classifier.

        :param x: Test set.
        :type x: `np.ndarray`
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :type kwargs: `dict`
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)


class StaticAdversarialTrainer(AdversarialTrainer):
    """
    Class performing adversarial training based on a model architecture and one or multiple attack methods. This method
    is static in the sense that all adversarial samples are generated statically before training. They are added to the
    original training sample. Training is then performed on the mixed dataset. Each attack specified when creating the
    instance is applied to all samples in the dataset, and only the successful ones (on the source model) are kept for
    data augmentation. This implies that for `m` attacks and a training set of size `n`, the final training set has a
    maximum size of `(n + 1) * m`.
    """
    def fit(self, x, y, **kwargs):
        """
        Apply static adversarial training to a :class:`Classifier`.

        :param x: Training set.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param kwargs: Parameters to be passed on to the `fit` function of the classifier.
        :type kwargs: `dict`
        :return: `None`
        """
        x_augmented = list(x.copy())
        y_augmented = list(y.copy())
        labels = np.argmax(y, axis=1)

        # Generate adversarial samples for each attack
        for i, attack in enumerate(self.attacks):
            if 'targeted' in attack.attack_params and attack.targeted:
                    raise NotImplementedError("Adversarial training with targeted attacks is \
                                               currently not implemented")

            logger.info('Generating adversarial samples from attack: %i/%i.', i, len(self.attacks))
            # Predict new labels for the adversarial samples generated
            x_adv = attack.generate(x, y=y)
            y_pred = np.argmax(attack.classifier.predict(x_adv), axis=1)
            selected = np.array(labels != y_pred)
            logger.info('%i successful samples generated.', len(selected))

            # Only add successful attacks to augmented dataset
            x_augmented.extend(list(x_adv[selected]))
            y_augmented.extend(list(y[selected]))

        # Fit the model with the extended dataset
        self.x_augmented = np.array(x_augmented)
        self.y_augmented = np.array(y_augmented)
        self.classifier.fit(self.x_augmented, self.y_augmented, **kwargs)
