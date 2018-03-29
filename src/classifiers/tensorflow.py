from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import random

from src.classifiers.classifier import Classifier


class TFClassifier(Classifier):
    """
    This class implements a classifier with the Tensorflow framework.
    """
    def __init__(self, clip_values, input_ph, logits, use_logits=True,
                 output_ph=None, train=None, loss=None, sess=None):
        """
        Initialization specificly for the Tensorflow-based implementation.
        :param clip_values: (min, max) values for inputs
        :param input_ph: the input placeholder
        :param logits: the logits layer
        :param use_logits: (bool) whether to use logits for computing gradients
        :param output_ph: the output placeholder
        :param train: the train tensor for fitting, including an optimizer
        :param loss: the loss function for which to compute gradients
        :param sess: tensorflow session
        """
        super().__init__(clip_values)
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
        Do prediction for the model.
        :param inputs:
        :return: the prediction results
        """
        preds = tf.nn.softmax(self._logits)
        results = self._sess.run(preds, feed_dict={self._input_ph: inputs})

        return results

    def fit(self, inputs, outputs, batch_size=128, nb_epochs=10):
        """
        Fit the model by training.
        :param inputs: the input images
        :param outputs: the outputs put directly into the output_ph phaceholder
        :param batch_size: mini batch size
        :param nb_epochs: number of epoches for training
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

    def nb_classes(self):
        """
        Return the number of output classes.
        :return: number of output classes
        """
        return self._nb_classes

    def class_gradient(self, input):
        """
        Compute the gradient of the logits/softmax layer wrt one input.
        :param input: one single input image
        :return: a numpy array containing gradient
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
        Compute the gradient of the loss function wrt one input.
        :param input: one single input image
        :param label: the ground true label of the input image
        :return: a numpy array containing gradient
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




