from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

# TODO Add tests for defences on classifier

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Classifier(ABC):
    """
    Base abstract class for all classifiers.
    """
    def __init__(self, defences=None, preprocessing=(0, 1)):
        """
        Initialize a :class:`Classifier` object.

        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        print('init Classifier')
        self._parse_defences(defences)

        if len(preprocessing) != 2:
            raise ValueError('`preprocessing` should be a tuple of 2 floats with the substract and divide values for'
                             'the model inputs.')
        self._preprocessing = preprocessing

    @abc.abstractmethod
    def predict(self, x, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
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

    @property
    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    @abc.abstractmethod
    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        raise NotImplementedError

    @abc.abstractmethod
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

    def _parse_defences(self, defences):
        self.defences = defences

        if defences:
            import re
            pattern = re.compile("featsqueeze[1-8]?")

            for d in defences:
                if pattern.match(d):
                    try:
                        from art.defences import FeatureSqueezing

                        bit_depth = int(d[-1])
                        self.feature_squeeze = FeatureSqueezing(bit_depth=bit_depth)
                    except:
                        raise ValueError('You must specify the bit depth for feature squeezing: featsqueeze[1-8]')

                # Add label smoothing
                if d == 'labsmooth':
                    from art.defences import LabelSmoothing
                    self.label_smooth = LabelSmoothing()

                # Add spatial smoothing
                if d == 'smooth':
                    from art.defences import SpatialSmoothing
                    self.smooth = SpatialSmoothing(channel_index=self.channel_index)

    def _apply_defences_fit(self, x, y):
        # Apply label smoothing if option is set
        if hasattr(self, 'label_smooth'):
            _, y = self.label_smooth(None, y)

        # Apply feature squeezing if option is set
        if hasattr(self, 'feature_squeeze'):
            x = self.feature_squeeze(x, clip_values=self.clip_values)

        return x, y

    def _apply_defences_predict(self, x):
        # Apply feature squeezing if option is set
        if hasattr(self, 'feature_squeeze'):
            x = self.feature_squeeze(x, clip_values=self.clip_values)

        # Apply inputs smoothing if option is set
        if hasattr(self, 'smooth'):
            x = self.smooth(x)

        return x

    def _apply_processing(self, x):
        import numpy as np

        sub, div = self._preprocessing
        sub = np.asarray(sub, dtype=x.dtype)
        div = np.asarray(div, dtype=x.dtype)

        res = x - sub
        res = res / div

        return res

    def _apply_processing_gradient(self, grad):
        import numpy as np

        _, div = self._preprocessing
        div = np.asarray(div, dtype=grad.dtype)
        res = grad / div
        return res


class ImageClassifier(Classifier):
    def __init__(self, clip_values, channel_index, defences=None, preprocessing=(0, 1)):
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
        print('init ImageClassifier')
        Classifier.__init__(self, defences=defences, preprocessing=preprocessing)

        if len(clip_values) != 2:
            raise ValueError('`clip_values` should be a tuple of 2 floats containing the allowed data range.')
        self._clip_values = clip_values

        self._channel_index = channel_index

    @property
    def input_shape(self):
        """
        Return the shape of one input.

        :return: Shape of one input for the classifier.
        :rtype: `tuple`
        """
        return self._input_shape

    @property
    def clip_values(self):
        """
        :return: Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
        :rtype: `tuple`
        """
        return self._clip_values

    @property
    def channel_index(self):
        """
        :return: Index of the axis in data containing the color channels or features.
        :rtype: `int`
        """
        return self._channel_index


class TextClassifier(Classifier):
    """
    This is the base abstract class for all text classification models. It is an extension of the base
    :class:`Classifier`. Its current form assumes that the text embedding is part of the classifier and that the inputs
    are word or character indices resulting from a mapping of text to indices. This mapping is supposed to be external
    of the classifier.
    """
    def __init__(self, defences=None, preprocessing=(0, 1)):
        """
        Initialize a :class:`TextClassifier` object.

        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        Classifier.__init__(self, defences=defences, preprocessing=preprocessing)

    def word_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. each word. The gradient of a word is computed as the L_1 norm
        of the loss gradients of its embedding.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of shape [batch_size, nb_words].
        :rtype: `np.ndarray`
        """
        import numpy as np

        grad = self.loss_gradient(x, y)
        grad = np.sum(np.abs(grad), axis=2)
        return grad

    def predict_from_embedding(self, x_emb, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs in embedding form.

        :param x_emb: Array of inputs in embedding form, often shaped as `(batch_size, input_length, embedding_size)`.
        :type x_emb: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def to_embedding(self, x):
        """
        Convert the received classifier input `x` from token (words or characters) indices to embeddings.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Embedding form of sample `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def to_id(self, x_emb, strategy='nearest', metric='cosine'):
        """
        Convert the received input from embedding space to classifier input (most often, token indices).

        :param x_emb: Array of inputs in embedding form, often shaped as `(batch_size, input_length, embedding_size)`.
        :type x_emb: `np.ndarray`
        :param strategy: Strategy from mapping from embedding space back to input space.
        :type strategy: `str` or `Callable`
        :param metric: Metric to be used in the embedding space when determining vocabulary token proximity.
        :type metric: `str` or `Callable`
        :return: Array of token indices for sample `x_emb`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError
