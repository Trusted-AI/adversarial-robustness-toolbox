from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random

from art.classifiers.classifier import Classifier


class PyTorchClassifier(Classifier):
    """
    This class implements a classifier with the PyTorch framework.
    """
    def __init__(self, clip_values, model, loss, input_shape, use_logits=False, defences=None):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: PyTorch model.
        :type model: `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training.
        :type loss: `torch.nn.modules.loss._Loss`
        :param input_shape: Shape of the input.
        :type input_shape: `tuple`
        :param use_logits: True if the output of the model are the logits
        :type use_logits: `bool`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        """
        import torch

        super(PyTorchClassifier, self).__init__(clip_values, defences)
        self._nb_classes = list(model.modules())[-1 if use_logits else -2].out_features
        self._input_shape = input_shape
        self._model = model
        self._loss = loss

    def predict(self, inputs, logits=False):
        """
        Perform prediction for a batch of inputs.

        :param inputs: Test set.
        :type inputs: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        # Set test phase
        self._model.train(False)

        # Run prediction
        if logits:
            results = self._sess.run(self._logits, feed_dict=fd)
        else:
            results = self._sess.run(tf.nn.softmax(self._logits), feed_dict=fd)

        return results

    def _forward_at(self, inputs, layer):
        """
        Compute the forward at a specific layer.

        :param inputs: Input data.
        :type inputs: `np.ndarray`
        :param layer: The layer where to get the forward results.
        :type layer: `int`
        :return: The forward results at the layer.
        :rtype: `np.ndarray`
        """

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
            raise ValueError("Need the training objective and the output placeholder to train the model.")

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
                if self._learning is None:
                    self._sess.run(self._train, feed_dict={self._input_ph: i_batch, self._output_ph: o_batch})
                else:
                    self._sess.run(self._train, feed_dict={self._input_ph: i_batch, self._output_ph: o_batch,
                                                           self._learning: True})

    def class_gradient(self, inputs, logits=False):
        """
        Compute per-class derivatives w.r.t. `input`.

        :param inputs: Sample input with shape as expected by the model.
        :type inputs: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)`.
        :rtype: `np.ndarray`
        """
        # Compute the gradient and return
        if logits:
            grds = self._sess.run(self._logit_class_grads, feed_dict={self._input_ph: inputs})
        else:
            grds = self._sess.run(self._class_grads, feed_dict={self._input_ph: inputs})

        grds = np.swapaxes(np.array(grds), 0, 1)

        return grds

    def loss_gradient(self, inputs, labels):
        """
        Compute the gradient of the loss function w.r.t. `inputs`.

        :param inputs: Sample input with shape as expected by the model.
        :type inputs: `np.ndarray`
        :param labels: Correct labels, one-vs-rest encoding.
        :type labels: `np.ndarray`
        :return: Array of gradients of the same shape as the inputs.
        :rtype: `np.ndarray`
        """
        # Check if loss available
        if not hasattr(self, '_loss_grads') or self._loss_grads is None:
            raise ValueError("Need the loss function to compute the loss gradient.")

        # Compute the gradient and return
        grds = self._sess.run(self._loss_grads, feed_dict={self._input_ph: inputs, self._output_ph: labels})

        return grds

