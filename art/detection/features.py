from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys
import logging
import six

import numpy as np

logger = logging.getLogger(__name__)

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Feature(ABC):
    """
    Base class for features. A feature object is defined for a classifier and enables feature-extraction for sample
    inputs. These features can be both layer-dependent or layer-independent. The extracted features can be
    used for training a detector.
    """

    def __init__(self, classifier):
        """
        Initialize a `Feature` object.

        :param classifier: Classification model for which the features will be extracted.
        :type classifier: :class:`.Classifier`
        """
        self.classifier = classifier

    @abc.abstractmethod
    def extract(self, x):
        """
        Extracts features for a set of inputs.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: extracted features for the inputs `x`.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        return NotImplementedError


class SaliencyMap(Feature):
    """
    Compute the saliency-map for a given sample `x` when classified by a classifier.
    Paper link: https://arxiv.org/abs/1312.6034
    """
    def __init__(self, classifier):
        """
        :param classifier: Classification model for which the features will be extracted.
        :type classifier: :class:`.Classifier`
        """
        super(SaliencyMap, self).__init__(classifier)

    def extract(self, x):
        """
        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Saliency map for the provided sample.
        :rtype: `np.ndarray`
        """
        return np.max(np.abs(self.classifier.class_gradient(x, label=None, logits=False)), axis=1)


class MeanClassDist(Feature):
    """
    Mean euclidean distances between features for a given layer.
    They are computed with respect to samples for different class from a given set of labelled images.
    """

    def __init__(self, classifier, x, y, layer=0, batch_size=32):
        """
        :param classifier: Classification model for which the features will be extracted.
        :type classifier: :class:`.Classifier`
        :param x: A set samples with respect to which the mean class distance is to be computed.
        :type x: `np.ndarray`
        :param y: Labels for the sample set x.
        :type y: `np.ndarray`
        :param layer: Layer for computing the features.
        :type layer: `int` or `str`
        :param batch_size: Batch size for computing activations.
        :type batch_size: `int`
        """

        super(MeanClassDist, self).__init__(classifier)
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
        layer_output = []

        for b in range(x.shape[0] // batch_size + 1):
            begin, end = b * batch_size, min((b + 1) * batch_size, x.shape[0])
            layer_output.append(self.classifier.get_activations(x[begin:end], self._layer_name))

        layer_output = np.concatenate(layer_output, axis=0)

        layer_output = layer_output.reshape(layer_output.shape[0], -1)

        assert y.shape[1] == classifier.nb_classes
        y_train = np.argmax(y, axis=1)

        # get layer output by classes
        self.layer_output_per_class = [layer_output[np.where(y_train == c)[0]]
                                       for c in range(self.classifier.nb_classes)]

    def extract(self, x):
        """
        Extracts features for a set of inputs.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Extracted features for the inputs `x`.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """

        layer_output = self.classifier.get_activations(x, self._layer_name)
        layer_output = layer_output.reshape(layer_output.shape[0], -1)

        dists_ = []
        norms2_x = np.sum(layer_output ** 2, 1)[:, None]

        for c in range(self.classifier.nb_classes):

            norms2_y = np.sum(self.layer_output_per_class[c] ** 2, 1)[None, :]
            pw_dists = norms2_x - 2 * np.matmul(layer_output, self.layer_output_per_class[c].T) + norms2_y
            dists_.append(np.mean(pw_dists, axis=1))

        return np.stack(dists_).T


class AttentionMap(Feature):
    """
    This feature estimates the pixels with highest influence on the prediction.
    """
    def __init__(self, classifier, window_width=8, strides=4):
        """
        :param classifier: Classification model for which the features will be extracted.
        :type classifier: :class:`.Classifier`
        :param window_width: Width of the grey-path window.
        :type window_width: `int`
        :param strides: Stride for the running window.
        :type strides: `int`
        """
        super(AttentionMap, self).__init__(classifier)
        self.window_width = window_width
        self.strides = strides

    def extract(self, x):
        """
        Extracts features for a set of inputs.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Extracted features for the inputs `x`.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        predictions = []
        for image in x:
            images = []
            for i in range(0, image.shape[0], self.strides):
                for j in range(0, image.shape[1], self.strides):
                    img = np.copy(image)
                    start_x = np.maximum(0, i - self.window_width + 1)
                    end_x = np.minimum(image.shape[0], i + self.window_width)
                    start_y = np.maximum(0, j - self.window_width + 1)
                    end_y = np.minimum(image.shape[1], j + self.window_width)
                    img[start_x:end_x, start_y:end_y, :] = 0.5
                    images.append(img)
            predictions.append(self.classifier.predict(np.array(images)))
        return np.array(predictions).reshape((x.shape[0], np.arange(0, image.shape[0], self.strides).shape[0],
                                              np.arange(0, image.shape[1], self.strides).shape[0], -1))


class KNNPreds(Feature):
    """
    K Nearest Neighbour prediction
    """
    def __init__(self, classifier, x, y, layer, batch_size=32, n_neighbors=50):
        """
        :param classifier: Classification model for which the features will be extracted.
        :type classifier: :class:`.Classifier`
        :param x: A set samples with respect to which the mean class distance is to be computed.
        :type x: `np.ndarray`
        :param y: Labels for the sample set `x`.
        :type y: `np.ndarray`
        :param layer: Layer for computing the features.
        :type layer: `int` or `str`
        :param batch_size: Batch size for computing activations.
        :type batch_size: `int`
        """
        from sklearn.neighbors import KNeighborsClassifier

        super(KNNPreds, self).__init__(classifier)

        if len(y.shape) > 1:
            y = y.ravel()

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

        layer_output = []

        for b in range(x.shape[0] // batch_size + 1):
            begin, end = b * batch_size, min((b + 1) * batch_size, x.shape[0])
            layer_output.append(self.classifier.get_activations(x[begin:end], self._layer_name))

        layer_output = np.concatenate(layer_output, axis=0)

        layer_output = layer_output.reshape(layer_output.shape[0], -1)

        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.neigh.fit(layer_output, y)

    def extract(self, x):
        """
        Extracts features for a set of inputs.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Extracted features for the inputs x.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """

        layer_output = self.classifier.get_activations(x, layer=self._layer_name)
        layer_output = layer_output.reshape(layer_output.shape[0], -1)

        return self.neigh.predict_proba(layer_output)
