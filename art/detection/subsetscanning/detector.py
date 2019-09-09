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
"""
This module implements the fast generalized subset scan based detector.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

# pylint: disable=E0001
import numpy as np
import six

from art.classifiers.classifier import Classifier, ClassifierNeuralNetwork, ClassifierGradients
from art.detection.subsetscanning.scanner import Scanner

logger = logging.getLogger(__name__)


class SubsetScanningDetector(ClassifierNeuralNetwork, ClassifierGradients, Classifier):
    """
    Fast generalized subset scan based detector by McFowland, E., Speakman, S., and Neill, D. B. (2013).

    | Paper link: https://www.cs.cmu.edu/~neill/papers/mcfowland13a.pdf
    """
    def __init__(self, classifier, bgd_data, layer):
        """
        Create a `SubsetScanningDetector` instance which is used to the detect the presence of adversarial samples.

        :param classifier: The model being evaluated for its robustness to anomalies (eg. adversarial samples)
        :type classifier: :class:`.Classifier`
        :bgd_data: The background data used to learn a null model. Typically dataset used to train the classifier.
        :type bgd_data: `np.ndarray`
        :layer: The layer from which to extract activations to perform scan
        :type layer: `int` or `str`
        """
        super(SubsetScanningDetector, self).__init__(clip_values=classifier.clip_values,
                                                     channel_index=classifier.channel_index,
                                                     defences=classifier.defences,
                                                     preprocessing=classifier.preprocessing)
        self.classifier = classifier
        self.bgd_data = bgd_data

        # Ensure that layer is well-defined:
        if isinstance(layer, six.string_types):
            if layer not in classifier.layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            self._layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(classifier.layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(classifier.layer_names) - 1))
            self._layer_name = classifier.layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        bgd_activations = classifier.get_activations(bgd_data, self._layer_name)
        if len(bgd_activations.shape) == 4:
            dim2 = bgd_activations.shape[1] * bgd_activations.shape[2] * bgd_activations.shape[3]
            bgd_activations = np.reshape(bgd_activations, (bgd_activations.shape[0], dim2))

        self.sorted_bgd_activations = np.sort(bgd_activations, axis=0)

    def calculate_pvalue_ranges(self, eval_x):
        """
        Returns computed p-value ranges.

        :param eval_x: Data being evaluated for anomalies.
        :type eval_x: `np.ndarray`
        :return: P-value ranges.
        :rtype: `np.ndarray`
        """

        bgd_activations = self.sorted_bgd_activations
        eval_activations = self.classifier.get_activations(eval_x, self._layer_name)

        if len(eval_activations.shape) == 4:
            dim2 = eval_activations.shape[1] * eval_activations.shape[2] * eval_activations.shape[3]
            eval_activations = np.reshape(eval_activations, (eval_activations.shape[0], dim2))

        bgrecords_n = bgd_activations.shape[0]
        records_n = eval_activations.shape[0]
        atrr_n = eval_activations.shape[1]

        pvalue_ranges = np.empty((records_n, atrr_n, 2))

        for j in range(atrr_n):
            pvalue_ranges[:, j, 0] = np.searchsorted(bgd_activations[:, j], eval_activations[:, j], side='right')
            pvalue_ranges[:, j, 1] = np.searchsorted(bgd_activations[:, j], eval_activations[:, j], side='left')

        pvalue_ranges = bgrecords_n - pvalue_ranges

        pvalue_ranges[:, :, 0] = np.divide(pvalue_ranges[:, :, 0], bgrecords_n + 1)
        pvalue_ranges[:, :, 1] = np.divide(pvalue_ranges[:, :, 1] + 1, bgrecords_n + 1)

        return pvalue_ranges

    def scan(self, clean_x, adv_x, clean_size=None, advs_size=None, run=10):
        """
        Returns scores of highest scoring subsets.

        :param clean_x: Data presumably without anomalies.
        :type clean_x: `np.ndarray`
        :param adv_x: Data presumably with anomalies (adversarial samples).
        :type adv_x: `np.ndarray`
        :param clean_size:
        :type clean_size: `int`
        :param advs_size:
        :param advs_size: `int`
        :return: (clean_scores, adv_scores, detectionpower)
        :rtype: `list`, `list`, `float`
        """
        from sklearn import metrics

        clean_pvalranges = self.calculate_pvalue_ranges(clean_x)
        adv_pvalranges = self.calculate_pvalue_ranges(adv_x)

        clean_scores = []
        adv_scores = []

        if clean_size is None and advs_size is None:

            # Individual scan
            for j, _ in enumerate(clean_pvalranges):
                best_score, _, _, _ = Scanner.fgss_individ_for_nets(clean_pvalranges[j])
                clean_scores.append(best_score)
            for j, _ in enumerate(adv_pvalranges):
                best_score, _, _, _ = Scanner.fgss_individ_for_nets(adv_pvalranges[j])
                adv_scores.append(best_score)

        else:

            len_adv_x = len(adv_x)
            len_clean_x = len(clean_x)

            for _ in range(run):
                np.random.seed()

                advchoice = np.random.choice(range(len_adv_x), advs_size, replace=False)
                cleanchoice = np.random.choice(range(len_clean_x), clean_size, replace=False)

                combined_pvals = np.concatenate((clean_pvalranges[cleanchoice], adv_pvalranges[advchoice]), axis=0)

                best_score, _, _, _ = Scanner.fgss_for_nets(clean_pvalranges[cleanchoice])
                clean_scores.append(best_score)
                best_score, _, _, _ = Scanner.fgss_for_nets(combined_pvals)
                adv_scores.append(best_score)

        y_true = np.append([np.ones(len(adv_scores))], [np.zeros(len(clean_scores))])
        all_scores = np.append([adv_scores], [clean_scores])

        fpr, tpr, _ = metrics.roc_curve(y_true, all_scores)
        roc_auc = metrics.auc(fpr, tpr)
        detectionpower = roc_auc

        return clean_scores, adv_scores, detectionpower

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the detector using training data.
        Assume that the classifier is already trained

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform detection of adversarial data and return prediction as tuple.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    def nb_classes(self):
        return self.detector.nb_classes()

    @property
    def input_shape(self):
        return self.detector.input_shape

    @property
    def clip_values(self):
        return self.detector.clip_values

    @property
    def channel_index(self):
        return self.detector.channel_index

    def learning_phase(self):
        return self.detector.learning_phase

    def class_gradient(self, x, label=None, **kwargs):
        return self.detector.class_gradient(x, label=label)

    def loss_gradient(self, x, y, **kwargs):
        return self.detector.loss_gradient(x, y)

    def get_activations(self, x, layer, batch_size):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`. This function is not supported for this detector.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    def set_learning_phase(self, train):
        self.detector.set_learning_phase(train)

    def save(self, filename, path=None):
        self.detector.save(filename, path)
