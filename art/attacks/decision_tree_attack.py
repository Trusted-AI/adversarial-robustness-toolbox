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

import abc
import sys

import numpy as np

logger = logging.getLogger(__name__)


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Decision_Tree_Attack(ABC):
    """
    # Close implementation of papernots attack on decision trees from https://arxiv.org/pdf/1605.07277.pdf , Algorithm 2. 
    # Implementation after Algorthim 2 and communication with the authors.  
    """
    attack_params = ['classifier','offset']

    def __init__(self, classifier,offset=0.001):
        """
        :param classifier: A trained model of type scikit decision tree.
        :type classifier: :class:`.Classifier.ScikitlearnDecisionTreeClassifier`
        :param offset: How much the value is changed away from tree's threshold
        :type classifier: :float:
        """
        self.classifier = classifier
        if not callable(getattr(classifier, 'get_decision_path')):
            raise TypeError('Model must be a decision tree model and provide a decision path!')
        if not callable(getattr(classifier, 'get_left_child')) or not callable(getattr(classifier, 'get_right_child')):
            raise TypeError('Model must be a decision tree model and provide children for nodes!')
        if not callable(getattr(classifier, 'get_feature_at_node')):
            raise TypeError('Model must be a decision tree model and provide a feature per node!')
        if not callable(getattr(classifier, 'get_threshold_at_node')):
            raise TypeError('Model must be a decision tree model and provide a threshold for each split!')
        if not callable(getattr(classifier, 'get_classes_at_node')):
            raise TypeError('Model must be a decision tree model and provide which class is classified at which node!')
        self.offset = offset
    
    def searchSubtree(self, position, original_class, target=None):
        #base case, we're at a leaf
        if self.classifier.get_left_child(position)==self.classifier.get_right_child(position):
            if target is None:
                if self.classifier.get_classes_at_node(position)!=original_class:
                    return [position]
                else:
                    return [-1]
            else:
                if self.classifier.get_classes_at_node(position) == target:
                    return [position]
                else:
                    return [-1]
        else: #go deeper, depths first
            res = self.searchSubtree(self.classifier.get_left_child(position), original_class, target)
            if res[0]==-1:
                #no result, try right subtree
                res = self.searchSubtree(self.classifier.get_right_child(position), original_class, target)  
                if res[0]==-1:
                    #no desired result
                    return [-1]
                else:
                    res.append(position)
                    return res
            else:
                #done, it is returning a path
                res.append(position)
                return res

    def generate(self, x, y=None):
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
        if y is not None:
            assert np.shape(y)[0]==np.shape(x)[0]
            if len(np.shape(y))>1:
                y = np.argmax(y)
        for index in range(np.shape(x)[0]):
            path = self.classifier.get_decision_path(x[index])
            legitimate_class = np.argmax(self.classifier.predict(x[index].reshape(1,-1)))
            position = -2
            adv_path = [-1]
            ancestor = path[position]
            while np.abs(position)< (len(path)-1) or adv_path[0]== -1:
                ancestor = path[position]
                current_child = path[position+1]
                if current_child == self.classifier.get_left_child(ancestor):  
                    if y is None:
                        adv_path = self.searchSubtree(self.classifier.get_right_child(ancestor),legitimate_class)
                    else:
                        adv_path = self.searchSubtree(self.classifier.get_right_child(ancestor),legitimate_class,y[index])
                else: #we are not the left leaf (binary tree)
                    if y is None:
                        adv_path = self.searchSubtree(self.classifier.get_left_child(ancestor),legitimate_class)
                    else:
                        adv_path = self.searchSubtree(self.classifier.get_left_child(ancestor),legitimate_class,y[index])
                position = position -1 #we are going the decision path updwards
            adv_path.append(ancestor)
            #print(adv_path,path)
            for i in range(1,1+len(adv_path[1:])): #first one is node, cannot be perturbed
                goFor = adv_path[i-1]
                threshold = self.classifier.get_threshold_at_node(adv_path[i])
                feature  = self.classifier.get_feature_at_node(adv_path[i])
                #print(goFor, threshold, x[index][feature],self.classifier.get_left_child(adv_path[i]),self.classifier.get_right_child(adv_path[i]))
                if x[index][feature]>threshold and goFor==self.classifier.get_left_child(adv_path[i]):
                    x[index][feature]=threshold-self.offset
                elif x[index][feature]<=threshold and goFor==self.classifier.get_right_child(adv_path[i]):
                    x[index][feature]=threshold+self.offset
        return x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True