from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random

from art.classifiers.classifier import Classifier


class PyTorchClassifier(Classifier):
    """
    This class implements a classifier with the PyTorch framework.
    """
    def __init__(self, clip_values, model, loss, optimizer, input_shape, use_logits=False, defences=None):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: PyTorch model.
        :type model: `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training.
        :type loss: `torch.nn.modules.loss._Loss`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param input_shape: Shape of the input.
        :type input_shape: `tuple`
        :param use_logits: True if the output of the model are the logits
        :type use_logits: `bool`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        """
        super(PyTorchClassifier, self).__init__(clip_values, defences)
        self._nb_classes = list(model.modules())[-1 if use_logits else -2].out_features
        self._input_shape = input_shape
        self._model = model
        self._loss = loss
        self._optimizer = optimizer

        # Store the logit layer
        self._logit_layer = len(list(model.modules())) - 2 if use_logits else len(list(model.modules())) - 3

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

        # Apply defences
        inputs = self._apply_defences_predict(inputs)

        # Run prediction
        preds = self._forward_at(inputs, self._logit_layer)
        if not logits:
            exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            preds = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

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
        # Set train phase
        self._model.train(True)

        # Apply defences
        inputs, outputs = self._apply_defences_fit(inputs, outputs)

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

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Actual training
                m_batch = self._model(i_batch)
                loss = self._loss(m_batch, o_batch)
                loss.backward()
                self._optimizer.step()

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
        results = inputs
        for l in list(self._model.modules())[1:layer + 2]:
            results = l(results)

        return results


