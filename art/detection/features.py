from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

import numpy as np
# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})



class Feature(ABC):
    """
    Base class for features

    Objects instantiated for this class can be used to extract features for a given classifier
    and a set of inages
    """

    def __init__(self, classifier):
        """

        :param classifier: ART classifier
        """
        self.classifier = classifier

    @abc.abstractmethod
    def extract(self, x):
        """

        :param x: Sample input with shape as expected by the model.
        :return: extracted features
        :rtype: `np.ndarray`
        """
        return NotImplementedError


class SaliencyMap(Feature):
    """
    Estimate the pixels with highest influence on the prediction.
    """
    def __init__(self, classifier):
        """
        :param classifier: ART classifier
        """
        super().__init__(classifier)

    def extract(self, x):
        """

        :param x: Sample input with shape as expected by the model.
        :return: returns saliencey map
        :rtype: `np.ndarray`
        """
        return np.max(np.abs(self.classifier.class_gradient(x, label=None,logits=False)), axis = 1)

class MeanClassDist(Feature):
    """
    Mean euclidean distances to samples for different class
    """
    def __init__(self,classifier,x,y,layerid=0,batch_size=32):
        """

        :param classifier: ART classifier
        :param x: a training set with respect to which the mean class distance is to be computed
        :param y: labels for the set x
        :param layerid: (optional) layer-id with respect to which the predictions are to be computed
        :param batch_size: (optional) batch_size for computing activations
        """

        super().__init__(classifier)
        self.layerid = layerid

        layer_output = []

        for b in range(x.shape[0] // batch_size + 1):
            begin, end = b * batch_size, min((b + 1) * batch_size, x.shape[0])
            layer_output.append(self.classifier.get_activations(x[begin:end], self.layerid))

        layer_output = np.concatenate(layer_output,axis=0)

        layer_output = layer_output.reshape(layer_output.shape[0],-1)

        assert y.shape[1] == classifier.nb_classes
        y_train = np.argmax(y,axis=1)

        # get layer output by classes
        self.layer_output_per_class = [layer_output[np.where(y_train == c)[0]]
                                  for c in range(self.classifier.nb_classes)]




    def extract(self,x):
        """
        :param x: Sample input with shape as expected by the model.
        :return: mean class distance for the activations at layer=layer_id
        :rtype: `np.ndarray`
        """

        layer_output = self.classifier.get_activations(x, self.layerid)
        layer_output = layer_output.reshape(layer_output.shape[0],-1)

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
    def __init__(self, classifier, window_width = 8, strides = 4):
        """

        :param classifier: ART classifier
        :param window_width: width of the grey-path window
        :param strides: stride for the runnning window
        """
        super().__init__(classifier)
        self.window_width = window_width
        self.strides = strides

    def extract(self, x):
        """
        :param x: Sample input with shape as expected by the model.
        :return: mean class distance for the activations at layer=layer_id
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
                    img[start_x:end_x,start_y:end_y,:] = 0.5
                    images.append(img)
            predictions.append(self.classifier.predict(np.array(images)))
        return np.array(predictions).reshape((x.shape[0], np.arange(0, image.shape[0], self.strides).shape[0], np.arange(0, image.shape[1], self.strides).shape[0], -1))

