# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements attacks on Decision Trees.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Union

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.utils import check_and_transform_label_format, compute_success

logger = logging.getLogger(__name__)


class DecisionTreeAttack(EvasionAttack):
    """
    Close implementation of Papernot's attack on decision trees following Algorithm 2 and communication with the
    authors.

    | Paper link: https://arxiv.org/abs/1605.07277
    """

    attack_params = ["classifier", "offset", "verbose"]
    _estimator_requirements = (ScikitlearnDecisionTreeClassifier,)

    def __init__(
        self,
        classifier: ScikitlearnDecisionTreeClassifier,
        offset: float = 0.001,
        verbose: bool = True,
    ) -> None:
        """
        :param classifier: A trained scikit-learn decision tree model.
        :param offset: How much the value is pushed away from tree's threshold.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.offset = offset
        self.verbose = verbose
        self._check_params()

    def _df_subtree(
        self,
        position: int,
        original_class: Union[int, np.ndarray],
        target: Optional[int] = None,
    ) -> List[int]:
        """
        Search a decision tree for a mis-classifying instance.

        :param position: An array with the original inputs to be attacked.
        :param original_class: original label for the instances we are searching mis-classification for.
        :param target: If the provided, specifies which output the leaf has to have to be accepted.
        :return: An array specifying the path to the leaf where the classification is either != original class or
                 ==target class if provided.
        """
        # base case, we're at a leaf
        if self.estimator.get_left_child(position) == self.estimator.get_right_child(position):
            if target is None:  # untargeted case
                if self.estimator.get_classes_at_node(position) != original_class:
                    path = [position]
                else:
                    path = [-1]
            else:  # targeted case
                if self.estimator.get_classes_at_node(position) == target:
                    path = [position]
                else:
                    path = [-1]
        else:  # go deeper, depths first
            res = self._df_subtree(self.estimator.get_left_child(position), original_class, target)
            if res[0] == -1:
                # no result, try right subtree
                res = self._df_subtree(self.estimator.get_right_child(position), original_class, target)
                if res[0] == -1:
                    # no desired result
                    path = [-1]
                else:
                    res.append(position)
                    path = res
            else:
                # done, it is returning a path
                res.append(position)
                path = res

        return path

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes, return_one_hot=False)
        x_adv = x.copy()

        for index in trange(x_adv.shape[0], desc="Decision tree attack", disable=not self.verbose):
            path = self.estimator.get_decision_path(x_adv[index])
            legitimate_class = np.argmax(self.estimator.predict(x_adv[index].reshape(1, -1)))
            position = -2
            adv_path = [-1]
            ancestor = path[position]
            while np.abs(position) < (len(path) - 1) or adv_path[0] == -1:
                ancestor = path[position]
                current_child = path[position + 1]
                # search in right subtree
                if current_child == self.estimator.get_left_child(ancestor):
                    if y is None:
                        adv_path = self._df_subtree(self.estimator.get_right_child(ancestor), legitimate_class)
                    else:
                        adv_path = self._df_subtree(
                            self.estimator.get_right_child(ancestor),
                            legitimate_class,
                            y[index],
                        )
                else:  # search in left subtree
                    if y is None:
                        adv_path = self._df_subtree(self.estimator.get_left_child(ancestor), legitimate_class)
                    else:
                        adv_path = self._df_subtree(
                            self.estimator.get_left_child(ancestor),
                            legitimate_class,
                            y[index],
                        )
                position = position - 1  # we are going the decision path upwards
            adv_path.append(ancestor)
            # we figured out which is the way to the target, now perturb
            # first one is leaf-> no threshold, cannot be perturbed
            for i in range(1, 1 + len(adv_path[1:])):
                go_for = adv_path[i - 1]
                threshold = self.estimator.get_threshold_at_node(adv_path[i])
                feature = self.estimator.get_feature_at_node(adv_path[i])
                # only perturb if the feature is actually wrong
                if x_adv[index][feature] > threshold and go_for == self.estimator.get_left_child(adv_path[i]):
                    x_adv[index][feature] = threshold - self.offset
                elif x_adv[index][feature] <= threshold and go_for == self.estimator.get_right_child(adv_path[i]):
                    x_adv[index][feature] = threshold + self.offset

        logger.info(
            "Success rate of decision tree attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv),
        )
        return x_adv

    def _check_params(self) -> None:
        if not isinstance(self.estimator, ScikitlearnDecisionTreeClassifier):
            raise TypeError("Model must be a decision tree model.")

        if self.offset <= 0:
            raise ValueError("The offset parameter must be strictly positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
