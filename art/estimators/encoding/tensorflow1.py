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
This module implements the classifier `TensorFlowEncoder` for TensorFlow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import tensorflow as tf

from art.estimators.tensorflow import TensorFlowEstimator
from art.estimators.encoding.encoder import EncoderMixin

logger = logging.getLogger(__name__)


class Tensorflow1InverseGan(EncoderMixin, TensorFlowEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements an inverse gan with the TensorFlow framework.
    """

    def __init__(
            self,
            input_ph,
            model,
            loss=None,
            sess=None,
            channel_index=3,
            clip_values=None,
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=(0, 1),
            feed_dict={}
    ):
        """
        Initialization specific to TensorFlow inverse gan implementations.

        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param model: tensorflow model, neural network or other.
        :type model: `tf.Tensor`
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
        model and when computing gradients w.r.t. the loss function.
        :type loss: `tf.Tensor`
        :param sess: Computation session.
        :type sess: `tf.Session`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param feed_dict: A feed dictionary for the session run evaluating the classifier. This dictionary includes all
                          additionally required placeholders except the placeholders defined in this class.
        :type feed_dict: `dictionary`
        """

        super(Tensorflow1InverseGan, self).__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._nb_classes = int(model.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        self._input_ph = input_ph
        self._model = model
        self._encoding_length = self._model.shape[1]
        self._loss = loss
        self._feed_dict = feed_dict

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
        self._sess = sess

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of encoding predictions of shape `(num_inputs, encoding_length)`.
        :rtype: `np.ndarray`
        """
        logger.info("Encoding input")
        y = self._sess.run(self._model, feed_dict={self._input_ph: x})
        return y

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        """
        do nothing.
        """
        pass

    def get_activations(self, x, layer, batch_size=128):
        """
        do nothing.
        """
        pass

    def set_learning_phase(self, train):
        """
         do nothing.
         """
        pass

    def loss_gradient(self, z_encoding, image_adv):
        """
        No gradients to compute for this method; do nothing.
        """
        pass

    @property
    def encoding_length(self):
        """
        Returns the length of the encoding size output
        :return:
        :rtype: `int`
        """
        return self._encoding_length
