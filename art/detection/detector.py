"""
Module containing different methods for the detection of adversarial examples. All models are considered to be binary
detectors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import six

from art.classifiers import Classifier

logger = logging.getLogger(__name__)


class BinaryInputDetector(Classifier):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and trains it on data labeled as clean (label 0) or adversarial (label 1).
    """
    def __init__(self, detector):
        """
        Create a `BinaryInputDetector` instance which performs binary classification on input data.

        :param detector: The detector architecture to be trained and applied for the binary classification.
        :type detector: :class:`.Classifier`
        """
        super(BinaryInputDetector, self).__init__(clip_values=detector.clip_values,
                                                  channel_index=detector.channel_index,
                                                  defences=detector.defences,
                                                  preprocessing=detector.preprocessing)
        self.detector = detector

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the detector using clean and adversarial samples.

        :param x: Training set to fit the detector.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        self.detector.fit(x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def predict(self, x, logits=False, batch_size=128):
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        return self.detector.predict(x, logits=logits, batch_size=batch_size)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    @property
    def nb_classes(self):
        return self.detector.nb_classes

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

    def class_gradient(self, x, label=None, logits=False):
        return self.detector.class_gradient(x, label=label, logits=logits)

    def loss_gradient(self, x, y):
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


class BinaryActivationDetector(Classifier):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and is trained on the values of the activations of a classifier at a given layer.
    """
    def __init__(self, classifier, detector, layer):
        """
        Create a `BinaryActivationDetector` instance which performs binary classification on activation information.
        The shape of the input of the detector has to match that of the output of the chosen layer.

        :param classifier: The classifier of which the activation information is to be used for detection.
        :type classifier: `art.classifier.Classifier`
        :param detector: The detector architecture to be trained and applied for the binary classification.
        :type detector: `art.classifier.Classifier`
        :param layer: Layer for computing the activations to use for training the detector.
        :type layer: `int` or `str`
        """
        super(BinaryActivationDetector, self).__init__(clip_values=detector.clip_values,
                                                       channel_index=detector.channel_index,
                                                       defences=detector.defences,
                                                       preprocessing=detector.preprocessing)
        self.classifier = classifier
        self.detector = detector

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

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the detector using training data.

        :param x: Training set to fit the detector.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        x_activations = self.classifier.get_activations(x, self._layer_name)
        self.detector.fit(x_activations, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def predict(self, x, logits=False, batch_size=128):
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        return self.detector.predict(self.classifier.get_activations(x, self._layer_name))

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    @property
    def nb_classes(self):
        return self.detector.nb_classes

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

    def class_gradient(self, x, label=None, logits=False):
        return self.detector.class_gradient(x, label=label, logits=logits)

    def loss_gradient(self, x, y):
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


class FeatureBasedDetector(Classifier):
    """
    A feature based detector.
    """
    def __init__(self, classifier, detector, feature, feature_params=None):
        """
        Create a `FeatureBasedDetector` instance which performs binary classification on activation information.
        The shape of the input of the detector has to match that of the output of the chosen layer.

        :param classifier: The classifier of which the feature information is to be used for detection.
        :type classifier: `art.classifier.Classifier`
        :param detector: The detector architecture to be trained and applied for the binary classification.
        :type detector: `art.classifier.Classifier`
        :param feature: The feature to be used for binary classification.
        :type feature: `art.detection.Feature`
        :param feature_params: Feature paramters to be used while instantiating the feature.
        :type feature_params: `dict`
        """
        super(FeatureBasedDetector, self).__init__(clip_values=detector.clip_values,
                                                   channel_index=detector.channel_index,
                                                   defences=detector.defences,
                                                   preprocessing=detector.preprocessing)
        self.classifier = classifier
        self.detector = detector
        self._feature = feature(classifier, **feature_params)

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the detector using training data.

        :param x: Training set to fit the detector.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        x_features = self._feature.extract(x)
        x_features = x_features.reshape(x_features.shape[0], -1)
        self.detector.fit(x_features, y, **kwargs)

    def predict(self, x, logits=False, batch_size=128):
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        x_features = self._feature.extract(x)
        x_features = x_features.reshape(x_features.shape[0], -1)
        return self.detector.predict(x_features, logits=logits, batch_size=batch_size)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    @property
    def nb_classes(self):
        return self.detector.nb_classes

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

    def class_gradient(self, x, label=None, logits=False):
        return self.detector.class_gradient(x, label=label, logits=logits)

    def loss_gradient(self, x, y):
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
