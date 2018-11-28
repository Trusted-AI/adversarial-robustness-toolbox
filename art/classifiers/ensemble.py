from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from art.classifiers import Classifier

class EnsembleClassifier(Classifier):
    """
    Base class for all classifiers.
    """
    def __init__(self, clip_values, *classifiers, classifier_weights=None, channel_index=3, defences=None, preprocessing=(0, 1), **kwargs):
        """
        Initialize a `Classifier` object.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(EnsembleClassifier, self).__init__(clip_values=clip_values, channel_index=channel_index, defences=defences,
 preprocessing=preprocessing)
        # assert len(classifiers) > 0
        self._nb_classes = classifiers[0].nb_classes
        # Assert all classifiers are the right shape(s)
        if classifier_weights is None:
            self.classifier_weights = np.ones(len(classifiers))
            self.classifier_weights = self.classifier_weights / sum(self.classifier_weights)
        self._classifier_weights = classifier_weights
        self._classifiers = classifiers
        self.__num_classifiers = len(self._classifiers)

    def predict(self, x, logits=False, raw=False):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param raw: Return the individual classifier raw outputs
        :type raw: `bool`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        preds = np.array([self.classifier_weights[i] * self._classifiers[i].predict(x,raw and logits) for i in range(self.__num_classifiers)])
        if raw:
            return preds
        z = np.sum(preds, axis=0)
        if logits:
            eps = 10e-8
            z = np.log(np.clip(z, eps, 1. - eps))# - np.log(np.clip(1. - z, eps, 1. - eps))
        return z

    def fit(self, x, y, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        raise NotImplementedError

    def get_activations(self, x, layer):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def class_gradient(self, x, label=None, logits=False, raw=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param raw: Return the individual classifier raw outputs
        :type raw: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        grads = np.array([self.classifier_weights[i] * self._classifiers[i].class_gradient(x, label, logits) for i in range(self.__num_classifiers)])
        if raw:
            return grads
        return np.sum(grads, axis=0)

    def loss_gradient(self, x, y, raw=False):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param raw: Return the individual classifier raw outputs
        :type raw: `bool`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        grads = np.array([self.classifier_weights[i] * self._classifiers[i].loss_gradient(x,y) for i in range(self.__num_classifiers)])
        if raw:
            return grads
        return np.sum(grads, axis=0)
