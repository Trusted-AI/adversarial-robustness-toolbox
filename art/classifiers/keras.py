from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from art.classifiers import Classifier


class KerasClassifier(Classifier):
    """
    The supported backends for Keras are TensorFlow and Theano.
    """
    def __init__(self, clip_values, model, use_logits=False, channel_index=3, defences=None):
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: Keras model
        :type model: `keras.models.Sequential`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        """
        import keras.backend as k

        # TODO Generalize loss function?
        super(KerasClassifier, self).__init__(clip_values, channel_index, defences)

        self._model = model
        self._input = model.input
        self._output = model.output
        _, self._nb_classes = k.int_shape(model.output)
        self._input_shape = k.int_shape(model.input)[1:]

        # Get predictions and loss function
        label_ph = k.placeholder(shape=(None,))
        if not use_logits:
            if k.backend() == 'tensorflow':
                preds, = self._output.op.inputs
                loss = k.sparse_categorical_crossentropy(label_ph, preds, from_logits=True)
            else:
                loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)

                # Convert predictions to logits for consistency with the other cases
                eps = 10e-8
                preds = k.log(k.clip(self._output, eps, 1. - eps))
        else:
            preds = self._output
            loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)
        loss_grads = k.gradients(loss, self._input)

        if k.backend() == 'tensorflow':
            loss_grads = loss_grads[0]
        elif k.backend() == 'cntk':
            raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

        # Set loss, grads and prediction functions
        self._preds_op = preds
        self._loss = k.function([self._input], [loss])
        self._loss_grads = k.function([self._input, label_ph], [loss_grads])
        self._preds = k.function([self._input], [preds])

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
        return self._loss_grads([x, np.argmax(y, axis=1)])[0]

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
        if logits:
            if not hasattr(self, '_class_grads_logits'):
                self._init_class_grads(logits=True)
            return np.swapaxes(np.array(self._class_grads_logits([x])), 0, 1)
        else:
            if not hasattr(self, '_class_grads'):
                self._init_class_grads(logits=False)
            return np.swapaxes(np.array(self._class_grads([x])), 0, 1)

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
        import keras.backend as k
        k.set_learning_phase(0)

        # Apply defences
        x = self._apply_defences_predict(x)

        # Run predictions with batching
        batch_size = 512
        preds = np.zeros((x.shape[0], self.nb_classes), dtype=np.float32)
        for b in range(x.shape[0] // batch_size + 1):
            begin, end = b * batch_size,  min((b + 1) * batch_size, x.shape[0])
            preds[begin:end] = self._preds([x[begin:end]])[0]

            if not logits:
                exp = np.exp(preds[begin:end] - np.max(preds[begin:end], axis=1, keepdims=True))
                preds[begin:end] = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def fit(self, x, y, batch_size=128, nb_epochs=20):
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
        import keras.backend as k
        k.set_learning_phase(1)

        # Apply defences
        x, y = self._apply_defences_fit(x, y)

        gen = generator_fit(x, y, batch_size)
        self._model.fit_generator(gen, steps_per_epoch=x.shape[0] / batch_size, epochs=nb_epochs)

    def _init_class_grads(self, logits=False):
        import keras.backend as k

        # Compute gradient per class, with and without the softmax activation
        if logits:
            class_grads_logits = [k.gradients(self._preds_op[:, i], self._input)[0] for i in range(self.nb_classes)]
            self._class_grads_logits = k.function([self._input], class_grads_logits)
        else:
            class_grads = [k.gradients(k.softmax(self._preds_op)[:, i], self._input)[0] for i in range(self.nb_classes)]
            self._class_grads = k.function([self._input], class_grads)


def generator_fit(x, y, batch_size=128):
    """
    Minimal data generator for randomly batching large datasets.

    :param x: The data sample to batch.
    :type x: `np.ndarray`
    :param y: The labels for `x`. The first dimension has to match the first dimension of `x`.
    :type y: `np.ndarray`
    :param batch_size: The size of the batches to produce.
    :type batch_size: `int`
    :return: A batch of size `batch_size` of random samples from `(x, y)`
    :rtype: `tuple(np.ndarray, np.ndarray)`
    """
    while True:
        indices = np.random.randint(x.shape[0], size=batch_size)
        yield x[indices], y[indices]
