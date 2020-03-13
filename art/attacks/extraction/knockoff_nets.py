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
This module implements the Knockoff Nets attack `KnockoffNets`.

| Paper link: https://arxiv.org/abs/1812.02766
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.classifiers.classifier import Classifier
from art.attacks.attack import ExtractionAttack
from art.utils import to_categorical


logger = logging.getLogger(__name__)


class KnockoffNets(ExtractionAttack):
    """
    Implementation of the Knockoff Nets attack from Orekondy et al. (2018).

    | Paper link: https://arxiv.org/abs/1812.02766
    """

    attack_params = ExtractionAttack.attack_params + [
        "batch_size_fit",
        "batch_size_query",
        "nb_epochs",
        "nb_stolen",
        "sampling_strategy",
        "reward",
    ]

    def __init__(
        self,
        classifier,
        batch_size_fit=1,
        batch_size_query=1,
        nb_epochs=10,
        nb_stolen=1,
        sampling_strategy="random",
        reward="all",
    ):
        """
        Create a KnockoffNets attack instance. Note, it is assumed that both the victim classifier and the thieved
        classifier produce logit outputs.

        :param classifier: A victim classifier.
        :type classifier: :class:`.Classifier`
        :param batch_size_fit: Size of batches for fitting the thieved classifier.
        :type batch_size_fit: `int`
        :param batch_size_query: Size of batches for querying the victim classifier.
        :type batch_size_query: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param nb_stolen: Number of queries submitted to the victim classifier to steal it.
        :type nb_stolen: `int`
        :param sampling_strategy: Sampling strategy, either `random` or `adaptive`.
        :type sampling_strategy: `string`
        :param reward: Reward type, in ['cert', 'div', 'loss', 'all'].
        :type reward: `string`
        """
        super(KnockoffNets, self).__init__(classifier=classifier)

        params = {
            "batch_size_fit": batch_size_fit,
            "batch_size_query": batch_size_query,
            "nb_epochs": nb_epochs,
            "nb_stolen": nb_stolen,
            "sampling_strategy": sampling_strategy,
            "reward": reward,
        }
        self.set_params(**params)

    def extract(self, x, y=None, **kwargs):
        """
        Extract a thieved classifier.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray` or `None`
        :param thieved_classifier: A thieved classifier to be stolen.
        :type thieved_classifier: :class:`.Classifier`
        :return: The stolen classifier.
        :rtype: :class:`.Classifier`
        """
        # Check prerequisite for random strategy
        if self.sampling_strategy == "random" and y is not None:
            logger.warning("This attack with random sampling strategy does not use the provided label y.")

        # Check prerequisite for adaptive strategy
        if self.sampling_strategy == "adaptive" and y is None:
            raise ValueError("This attack with adaptive sampling strategy needs label y.")

        # Check the size of the source input vs nb_stolen
        if x.shape[0] < self.nb_stolen:
            logger.warning(
                "The size of the source input is smaller than the expected number of queries submitted "
                "to the victim classifier."
            )

        # Check if there is a thieved classifier provided for training
        thieved_classifier = kwargs.get("thieved_classifier")
        if thieved_classifier is None or not isinstance(thieved_classifier, Classifier):
            raise ValueError("A thieved classifier is needed.")

        # Implement model extractions
        if self.sampling_strategy == "random":
            thieved_classifier = self._random_extraction(x, thieved_classifier)
        else:
            thieved_classifier = self._adaptive_extraction(x, y, thieved_classifier)

        return thieved_classifier

    def _random_extraction(self, x, thieved_classifier):
        """
        Extract with the random sampling strategy.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :param thieved_classifier: A thieved classifier to be stolen.
        :type thieved_classifier: :class:`.Classifier`
        :return: The stolen classifier.
        :rtype: :class:`.Classifier`
        """
        # Select data to attack
        selected_x = self._select_data(x)

        # Query the victim classifier
        fake_labels = self._query_label(selected_x)

        # Train the thieved classifier
        thieved_classifier.fit(
            x=selected_x, y=fake_labels, batch_size=self.batch_size_fit, nb_epochs=self.nb_epochs, verbose=0
        )

        return thieved_classifier

    def _select_data(self, x):
        """
        Select data to attack.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :return: An array with the selected input to the victim classifier.
        :rtype: `np.ndarray`
        """
        nb_stolen = np.minimum(self.nb_stolen, x.shape[0])
        rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)

        return x[rnd_index].astype(ART_NUMPY_DTYPE)

    def _query_label(self, x):
        """
        Query the victim classifier.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :return: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :rtype: `np.ndarray`
        """
        labels = self.classifier.predict(x=x, batch_size=self.batch_size_query)
        labels = np.argmax(labels, axis=1)
        labels = to_categorical(labels=labels, nb_classes=self.classifier.nb_classes())

        return labels

    def _adaptive_extraction(self, x, y, thieved_classifier):
        """
        Extract with the adaptive sampling strategy.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param thieved_classifier: A thieved classifier to be stolen.
        :type thieved_classifier: :class:`.Classifier`
        :return: The stolen classifier.
        :rtype: :class:`.Classifier`
        """
        # Compute number of actions
        if len(y.shape) == 2:
            nb_actions = len(np.unique(np.argmax(y, axis=1)))
        elif len(y.shape) == 1:
            nb_actions = len(np.unique(y))
        else:
            raise ValueError("Target values `y` has a wrong shape.")

        # We need to keep an average version of the victim output
        if self.reward == "div" or self.reward == "all":
            self.y_avg = np.zeros(self.classifier.nb_classes())

        # We need to keep an average and variance version of rewards
        if self.reward == "all":
            self.reward_avg = np.zeros(3)
            self.reward_var = np.zeros(3)

        # Implement the bandit gradients algorithm
        h_func = np.zeros(nb_actions)
        learning_rate = np.zeros(nb_actions)
        probs = np.ones(nb_actions) / nb_actions
        selected_x = []
        queried_labels = []

        avg_reward = 0
        for it in range(1, self.nb_stolen + 1):
            # Sample an action
            action = np.random.choice(np.arange(0, nb_actions), p=probs)

            # Sample data to attack
            sampled_x = self._sample_data(x, y, action)
            selected_x.append(sampled_x)

            # Query the victim classifier
            y_output = self.classifier.predict(x=np.array([sampled_x]), batch_size=self.batch_size_query)
            fake_label = np.argmax(y_output, axis=1)
            fake_label = to_categorical(labels=fake_label, nb_classes=self.classifier.nb_classes())
            queried_labels.append(fake_label[0])

            # Train the thieved classifier
            thieved_classifier.fit(
                x=np.array([sampled_x]), y=fake_label, batch_size=self.batch_size_fit, nb_epochs=1, verbose=0
            )

            # Test new labels
            y_hat = thieved_classifier.predict(x=np.array([sampled_x]), batch_size=self.batch_size_query)

            # Compute rewards
            reward = self._reward(y_output, y_hat, it)
            avg_reward = avg_reward + (1.0 / it) * (reward - avg_reward)

            # Update learning rate
            learning_rate[action] += 1

            # Update H function
            for a in range(nb_actions):
                if a != action:
                    h_func[a] = h_func[a] - 1.0 / learning_rate[action] * (reward - avg_reward) * probs[a]
                else:
                    h_func[a] = h_func[a] + 1.0 / learning_rate[action] * (reward - avg_reward) * (1 - probs[a])

            # Update probs
            aux_exp = np.exp(h_func)
            probs = aux_exp / np.sum(aux_exp)

        # Train the thieved classifier the final time
        thieved_classifier.fit(
            x=np.array(selected_x), y=np.array(queried_labels), batch_size=self.batch_size_fit, nb_epochs=self.nb_epochs
        )

        return thieved_classifier

    @staticmethod
    def _sample_data(x, y, action):
        """
        Sample data with a specific action.

        :param x: An array with the source input to the victim classifier.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param action: The action index returned from the action sampling.
        :type action: `int`
        :return: An array with one input to the victim classifier.
        :rtype: `np.ndarray`
        """
        if len(y.shape) == 2:
            y_ = np.argmax(y, axis=1)
        else:
            y_ = y

        x_ = x[y_ == action]
        rnd_idx = np.random.choice(len(x_))

        return x_[rnd_idx]

    def _reward(self, y_output, y_hat, n):
        """
        Compute reward value.

        :param y_output: Output of the victim classifier.
        :type y_output: `np.ndarray`
        :param y_hat: Output of the thieved classifier.
        :type y_hat: `np.ndarray`
        :param n: Current iteration.
        :type n: `int`
        :return: Reward value.
        :rtype: `float`
        """
        if self.reward == "cert":
            return self._reward_cert(y_output)
        elif self.reward == "div":
            return self._reward_div(y_output, n)
        elif self.reward == "loss":
            return self._reward_loss(y_output, y_hat)
        else:
            return self._reward_all(y_output, y_hat, n)

    @staticmethod
    def _reward_cert(y_output):
        """
        Compute `cert` reward value.

        :param y_output: Output of the victim classifier.
        :type y_output: `np.ndarray`
        :return: Reward value.
        :rtype: `float`
        """
        largests = np.partition(y_output.flatten(), -2)[-2:]
        reward = largests[1] - largests[0]

        return reward

    def _reward_div(self, y_output, n):
        """
        Compute `div` reward value.

        :param y_output: Output of the victim classifier.
        :type y_output: `np.ndarray`
        :param n: Current iteration.
        :type n: `int`
        :return: Reward value.
        :rtype: `float`
        """
        # First update y_avg
        self.y_avg = self.y_avg + (1.0 / n) * (y_output[0] - self.y_avg)

        # Then compute reward
        reward = 0
        for k in range(self.classifier.nb_classes()):
            reward += np.maximum(0, y_output[0][k] - self.y_avg[k])

        return reward

    def _reward_loss(self, y_output, y_hat):
        """
        Compute `loss` reward value.

        :param y_output: Output of the victim classifier.
        :type y_output: `np.ndarray`
        :param y_hat: Output of the thieved classifier.
        :type y_hat: `np.ndarray`
        :return: Reward value.
        :rtype: `float`
        """
        # Compute victim probs
        aux_exp = np.exp(y_output[0])
        probs_output = aux_exp / np.sum(aux_exp)

        # Compute thieved probs
        aux_exp = np.exp(y_hat[0])
        probs_hat = aux_exp / np.sum(aux_exp)

        # Compute reward
        reward = 0
        for k in range(self.classifier.nb_classes()):
            reward += -probs_output[k] * np.log(probs_hat[k])

        return reward

    def _reward_all(self, y_output, y_hat, n):
        """
        Compute `all` reward value.

        :param y_output: Output of the victim classifier.
        :type y_output: `np.ndarray`
        :param y_hat: Output of the thieved classifier.
        :type y_hat: `np.ndarray`
        :param n: Current iteration.
        :type n: `int`
        :return: Reward value.
        :rtype: `float`
        """
        reward_cert = self._reward_cert(y_output)
        reward_div = self._reward_div(y_output, n)
        reward_loss = self._reward_loss(y_output, y_hat)
        reward = [reward_cert, reward_div, reward_loss]
        self.reward_avg = self.reward_avg + (1.0 / n) * (reward - self.reward_avg)
        self.reward_var = self.reward_var + (1.0 / n) * ((reward - self.reward_avg) ** 2 - self.reward_var)

        # Normalize rewards
        if n > 1:
            reward = (reward - self.reward_avg) / np.sqrt(self.reward_var)
        else:
            reward = [max(min(r, 1), 0) for r in reward]

        return np.mean(reward)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param batch_size_fit: Size of batches for fitting the thieved classifier.
        :type batch_size_fit: `int`
        :param batch_size_query: Size of batches for querying the victim classifier.
        :type batch_size_query: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param nb_stolen: Number of queries submitted to the victim classifier to steal it.
        :type nb_stolen: `int`
        :param sampling_strategy: Sampling strategy, either `random` or `adaptive`.
        :type sampling_strategy: `string`
        :param reward: Reward type, in ['cert', 'div', 'loss', 'all'].
        :type reward: `string`
        """
        # Save attack-specific parameters
        super(KnockoffNets, self).set_params(**kwargs)

        if not isinstance(self.batch_size_fit, (int, np.int)) or self.batch_size_fit <= 0:
            raise ValueError("The size of batches for fitting the thieved classifier must be a positive integer.")

        if not isinstance(self.batch_size_query, (int, np.int)) or self.batch_size_query <= 0:
            raise ValueError("The size of batches for querying the victim classifier must be a positive integer.")

        if not isinstance(self.nb_epochs, (int, np.int)) or self.nb_epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")

        if not isinstance(self.nb_stolen, (int, np.int)) or self.nb_stolen <= 0:
            raise ValueError("The number of queries submitted to the victim classifier must be a positive integer.")

        if self.sampling_strategy not in ["random", "adaptive"]:
            raise ValueError("Sampling strategy must be either `random` or `adaptive`.")

        if self.reward not in ["cert", "div", "loss", "all"]:
            raise ValueError("Reward type must be in ['cert', 'div', 'loss', 'all'].")

        return True
