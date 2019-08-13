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
from art.attacks.attack import Attack
from art.classifiers.scikitklearn import ScikitlearnDecisionTreeClassifier
logger = logging.getLogger(__name__)


class DecisionTreeAttack(Attack):
    '''
        Close Implementation of the Decision Tree adversarial example formulation by Papernot et al. (2016), algorithm 2.
        Paper link:
        https://arxiv.org/pdf/1605.07277.pdf
    '''
    attack_params = ['classifier', 'offset']

    def __init__(self, classifier, offset=0.001):
        """
        :param classifier: A trained model of type scikit decision tree.
        :type classifier: :class:`.Classifier.ScikitlearnDecisionTreeClassifier`
        :param offset: How much the value is pushed away from tree's threshold. default 0.001
        :type classifier: :float:
        """
        super(DecisionTreeAttack, self).__init__(classifier=classifier)
        if not isinstance(classifier, ScikitlearnDecisionTreeClassifier):
            raise TypeError('Model must be a decision tree model!')
        params = {'offset': offset}
        self.set_params(**params)

    def _DFSubtree(self, position, original_class, target=None):
        """
        Search a decision tree for a misclassifying instance.

        :param position: An array with the original inputs to be attacked.
        :type position: `int`
        :param original_class: original label for the instances we are serching misclassification for.
        :type original_class: `int`
        :param target: If the provided, specifies which output the leaf has to have to be accepted. 
        :type target: `int`
        :return: An array specifying the path to the leaf where the classification is either != original class or 
               ==target class if provided.
        :rtype: `np.ndarray`
        """
        # base case, we're at a leaf
        if self.classifier.get_left_child(position) == self.classifier.get_right_child(position):
            if target is None:  # untargeted case
                if self.classifier.get_classes_at_node(position) != original_class:
                    return [position]
                else:
                    return [-1]
            else:  # targeted case
                if self.classifier.get_classes_at_node(position) == target:
                    return [position]
                else:
                    return [-1]
        else:  # go deeper, depths first
            res = self._DFSubtree(self.classifier.get_left_child(
                position), original_class, target)
            if res[0] == -1:
                # no result, try right subtree
                res = self._DFSubtree(self.classifier.get_right_child(
                    position), original_class, target)
                if res[0] == -1:
                    # no desired result
                    return [-1]
                else:
                    res.append(position)
                    return res
            else:
                # done, it is returning a path
                res.append(position)
                return res

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x = x.copy()
        if y is not None:
            assert np.shape(y)[0] == np.shape(x)[0]
            if len(np.shape(y)) > 1:
                y = np.argmax(y)
        for index in range(np.shape(x)[0]):
            path = self.classifier.get_decision_path(x[index])
            legitimate_class = np.argmax(
                self.classifier.predict(x[index].reshape(1, -1)))
            position = -2
            adv_path = [-1]
            ancestor = path[position]
            while np.abs(position) < (len(path)-1) or adv_path[0] == -1:
                ancestor = path[position]
                current_child = path[position+1]
                # serach in right subtree
                if current_child == self.classifier.get_left_child(ancestor):
                    if y is None:
                        adv_path = self._DFSubtree(
                            self.classifier.get_right_child(ancestor), legitimate_class)
                    else:
                        adv_path = self._DFSubtree(self.classifier.get_right_child(
                            ancestor), legitimate_class, y[index])
                else:  # serach in left subtree
                    if y is None:
                        adv_path = self._DFSubtree(
                            self.classifier.get_left_child(ancestor), legitimate_class)
                    else:
                        adv_path = self._DFSubtree(self.classifier.get_left_child(
                            ancestor), legitimate_class, y[index])
                position = position - 1  # we are going the decision path updwards
            adv_path.append(ancestor)
            # we figured out which is the way to the target, now perturb
            # first one is leaf-> no threshold, cannot be perturbed
            for i in range(1, 1+len(adv_path[1:])):
                goFor = adv_path[i-1]
                threshold = self.classifier.get_threshold_at_node(adv_path[i])
                feature = self.classifier.get_feature_at_node(adv_path[i])
                # only perturb if the feature is acutally wrong
                if x[index][feature] > threshold and goFor == self.classifier.get_left_child(adv_path[i]):
                    x[index][feature] = threshold-self.offset
                elif x[index][feature] <= threshold and goFor == self.classifier.get_right_child(adv_path[i]):
                    x[index][feature] = threshold+self.offset
        return x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        super(DecisionTreeAttack, self).set_params(**kwargs)

        if self.offset <= 0:
            raise ValueError("The offset parameter must be strictly positive.")
