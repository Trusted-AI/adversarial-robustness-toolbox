from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random

from art.classifiers import Classifier


class TFClassifier(Classifier):
    """
    This class implements a classifier with the Tensorflow framework.
    """
    def __init__(self, clip_values, input_ph, logits, output_ph=None, train=None, loss=None, learning=None, sess=None,
                 channel_index=3, defences=None):
        """
        Initialization specifically for the Tensorflow-based implementation.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param logits: The logits layer of the model.
        :type logits: `tf.Tensor`
        :param output_ph: The output layer of the model. Use this parameter only of `use_logits` is `False`.
        :type output_ph: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients.
        :type loss: `tf.Tensor`
        :param learning: The placeholder to indicate if the model is training.
        :type learning: `tf.Placeholder` of type bool.
        :param sess: Computation session.
        :type sess: `tf.Session`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        """
        import tensorflow as tf

        super(TFClassifier, self).__init__(clip_values, channel_index, defences)
        self._nb_classes = int(logits.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape()[1:])
        self._input_ph = input_ph
        self._logits = logits
        self._output_ph = output_ph
        self._train = train
        self._loss = loss
        self._learning = learning

        # Assign session
        if sess is None:
            self._sess = tf.get_default_session()
        else:
            self._sess = sess

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

    def predict(self, x, logits=False):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        import tensorflow as tf

        # Apply defences
        x = self._apply_defences_predict(x)

        # Create feed_dict
        fd = {self._input_ph: x}
        if self._learning is not None:
            fd[self._learning] = False

        # Run prediction
        if logits:
            results = self._sess.run(self._logits, feed_dict=fd)
        else:
            results = self._sess.run(tf.nn.softmax(self._logits), feed_dict=fd)

        return results

    def fit(self, x, y, batch_size=128, nb_epochs=10):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        # Check if train and output_ph available
        if self._train is None or self._output_ph is None:
            raise ValueError("Need the training objective and the output placeholder to train the model.")

        # Apply defences
        x, y = self._apply_defences_fit(x, y)

        num_batch = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                if m < num_batch - 1:
                    i_batch = x[ind[m * batch_size:(m + 1) * batch_size]]
                    o_batch = y[ind[m * batch_size:(m + 1) * batch_size]]
                else:
                    i_batch = x[ind[m * batch_size:]]
                    o_batch = y[ind[m * batch_size:]]

                # Run train step
                if self._learning is None:
                    self._sess.run(self._train, feed_dict={self._input_ph: i_batch, self._output_ph: o_batch})
                else:
                    self._sess.run(self._train, feed_dict={self._input_ph: i_batch, self._output_ph: o_batch,
                                                           self._learning: True})

    def class_gradient(self, x, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)`.
        :rtype: `np.ndarray`
        """
        # Compute the gradient and return
        if logits:
            if not hasattr(self, '_logit_class_grads'):
                self._init_class_grads(logits=True)
            grds = self._sess.run(self._logit_class_grads, feed_dict={self._input_ph: x})
        else:
            if not hasattr(self, '_class_grads'):
                self._init_class_grads(logits=False)
            grds = self._sess.run(self._class_grads, feed_dict={self._input_ph: x})

        grds = np.swapaxes(np.array(grds), 0, 1)

        return grds

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
        # Check if loss available
        if not hasattr(self, '_loss_grads') or self._loss_grads is None:
            raise ValueError("Need the loss function to compute the loss gradient.")

        # Compute the gradient and return
        grds = self._sess.run(self._loss_grads, feed_dict={self._input_ph: x, self._output_ph: y})

        return grds

    def _init_class_grads(self, logits=False):
        import tensorflow as tf

        # Construct the class gradients graph
        if logits:
            self._logit_class_grads = [tf.gradients(self._logits[:, i], self._input_ph)[0]
                                       for i in range(self._nb_classes)]
        else:
            self._class_grads = [tf.gradients(tf.nn.softmax(self._logits)[:, i], self._input_ph)[0]
                                 for i in range(self._nb_classes)]
