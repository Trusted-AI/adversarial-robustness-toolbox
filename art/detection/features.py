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


