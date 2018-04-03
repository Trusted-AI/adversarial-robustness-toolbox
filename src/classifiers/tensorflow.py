from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random

from src.classifiers.classifier import Classifier


class TFClassifier(Classifier):
    """
    This class implements a classifier with the Tensorflow framework.
    """
    def __init__(self, clip_values, input_ph, logits, use_logits=True, output_ph=None, train=None, loss=None,
                 sess=None):
        """
        Initialization specifically for the Tensorflow-based implementation.

         :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param logits: The logits layer of the model.
        :type logits: `tf.Tensor`
        :param use_logits: `True` if the output of the model are the logits.
        :type use_logits: `bool`
        :param output_ph: The output layer of the model. Use this parameter only of `use_logits` is `False`.
        :type output_ph: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients.
        :type loss: `tf.Tensor`
        :param sess: Computation session.
        :type sess: `tf.Session`
        """
        super(TFClassifier, self).__init__(clip_values)
        self._nb_classes = logits.get_shape()[-1]
        self._input_ph = input_ph
        self._logits = logits
        self._use_logits = use_logits
        self._output_ph = output_ph
        self._train = train
        self._loss = loss

        if sess is None:
            self._sess = tf.get_default_session()
        else:
            self._sess = sess

    def predict(self, inputs):
        """
        Perform prediction for a batch of inputs.

        :param inputs: Test set.
        :type inputs: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        preds = tf.nn.softmax(self._logits)
        results = self._sess.run(preds, feed_dict={self._input_ph: inputs})

        return results

    def fit(self, inputs, outputs, batch_size=128, nb_epochs=10):
        """
        Fit the classifier on the training set `(inputs, outputs)`.

        :param inputs: Training data.
        :type inputs: `np.ndarray`
        :param outputs: Labels.
        :type outputs: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        # Check if train and output_ph available
        if self._train is None or self._output_ph is None:
            raise ValueError("Need the train and the output placeholder to"
                             " train the model")

        num_batch = int(np.ceil(len(inputs) / batch_size))
        ind = np.arange(len(inputs))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                if m < num_batch - 1:
                    i_batch = inputs[ind[m * batch_size:(m + 1) * batch_size]]
                    o_batch = outputs[ind[m * batch_size:(m + 1) * batch_size]]
                else:
                    i_batch = inputs[ind[m*batch_size:]]
                    o_batch = outputs[ind[m * batch_size:]]

                # Run train step
                self._sess.run(self._train, feed_dict={
                    self._input_ph:i_batch, self._output_ph: o_batch})

    def class_gradient(self, input):
        """
        Compute per-class derivatives w.r.t. `input`.

        :param input: One sample input with shape as expected by the model.
        :type input: `np.ndarray`
        :return: Array of gradients of input features w.r.t. each class in the form `(self.nb_classes, input_shape)`
        :rtype: `np.ndarray`
        """
        # Get the function for the derivatives
        if not self._use_logits:
            preds = tf.nn.softmax(self._logits)
        else:
            preds = self._logits

        # Get the gradient graph
        grads = [tf.squeeze(tf.gradients(preds[:, i], self._input_ph)[0], 0)
                 for i in range(self._nb_classes)]

        # Compute the gradient and return
        grds = self._sess.run(grads, feed_dict={
            self._input_ph: np.array([input])})
        grds = np.array([g for g in grds])

        return grds

    def loss_gradient(self, input, label):
        """
        Compute the gradient of the loss function w.r.t. `input`.

        :param input: One sample input with shape as expected by the model.
        :type input: `np.ndarray`
        :param label: Correct label.
        :type label: `int`
        :return: Array of gradients of the same shape as `input`.
        :rtype: `np.ndarray`
        """
        # Check if loss available
        if self._loss is None:
            raise ValueError("Need the loss function to compute gradient")

        # Get the gradient graph
        grads = tf.gradients(self._loss, self._input_ph)[0]

        # Compute the gradient and return
        [grds] = self._sess.run(grads, feed_dict={
            self._input_ph: np.array([input]),
            self._output_ph: np.array([label])})

        return grds
