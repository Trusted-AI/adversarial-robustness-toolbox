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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import six

from art.classifiers import Classifier

logger = logging.getLogger(__name__)


class MXClassifier(Classifier):
    """
    Wrapper class for importing MXNet Gluon model.
    """
    def __init__(self, clip_values, model, input_shape, nb_classes, optimizer=None, ctx=None, channel_index=1,
                 defences=None, preprocessing=(0, 1)):
        """
        Initialize an `MXClassifier` object. Assumes the `model` passed as parameter is a Gluon model and that the
        loss function is the softmax cross-entropy.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: The model with logits as expected output.
        :type model: `mxnet.gluon.Block`
        :param input_shape: The shape of one input instance.
        :type input_shape: `tuple`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        :param optimizer: The optimizer used to train the classifier. This parameter is not required if no training is
               used.
        :type optimizer: `mxnet.gluon.Trainer`
        :param ctx: The device on which the model runs (CPU or GPU). If not provided, CPU is assumed.
        :type ctx: `mxnet.context.Context`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        import mxnet as mx

        super(MXClassifier, self).__init__(clip_values=clip_values, channel_index=channel_index, defences=defences,
                                           preprocessing=preprocessing)

        self._model = model
        self._nb_classes = nb_classes
        self._input_shape = input_shape
        self._device = ctx
        self._optimizer = optimizer

        if ctx is None:
            self._ctx = mx.cpu()
        else:
            self._ctx = ctx

        # Get the internal layer
        self._layer_names = self._get_layers()

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier on the training set `(inputs, outputs)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for MXNet
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        if self._optimizer is None:
            raise ValueError('An MXNet optimizer is required for fitting the model.')

        import mxnet as mx

        train_mode = self._learning_phase if hasattr(self, '_learning_phase') else True

        # Apply preprocessing and defences
        x_preproc = self._apply_processing(x)
        x_preproc, y_preproc = self._apply_defences(x_preproc, y, fit=True)
        y_preproc = np.argmax(y_preproc, axis=1)

        nb_batch = int(np.ceil(len(x_preproc) / batch_size))
        ind = np.arange(len(x_preproc))

        for _ in range(nb_epochs):
            # Shuffle the examples
            np.random.shuffle(ind)

            # Train for one epoch
            for m in range(nb_batch):
                x_batch = mx.nd.array(x_preproc[ind[m * batch_size:(m + 1) * batch_size]]).as_in_context(self._ctx)
                y_batch = mx.nd.array(y_preproc[ind[m * batch_size:(m + 1) * batch_size]]).as_in_context(self._ctx)

                with mx.autograd.record(train_mode=train_mode):
                    preds = self._model(x_batch)
                    loss = mx.nd.softmax_cross_entropy(preds, y_batch)
                loss.backward()

                # Update parameters
                self._optimizer.step(batch_size)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for MXNet
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        import mxnet as mx
        from art.data_generators import MXDataGenerator

        train_mode = self._learning_phase if hasattr(self, '_learning_phase') else True

        if isinstance(generator, MXDataGenerator) and \
                not (hasattr(self, 'label_smooth') or hasattr(self, 'feature_squeeze')):
            # Train directly in MXNet
            for _ in range(nb_epochs):
                for x_batch, y_batch in generator.data_loader:
                    x_batch = mx.nd.array(x_batch).as_in_context(self._ctx)
                    y_batch = mx.nd.argmax(y_batch, axis=1)
                    y_batch = mx.nd.array(y_batch).as_in_context(self._ctx)

                    with mx.autograd.record(train_mode=train_mode):
                        preds = self._model(x_batch)
                        loss = mx.nd.softmax_cross_entropy(preds, y_batch)
                    loss.backward()

                    # Update parameters
                    self._optimizer.step(x_batch.shape[0])
        else:
            # Fit a generic data generator through the API
            super(MXClassifier, self).fit_generator(generator, nb_epochs=nb_epochs)

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
        import mxnet as mx

        train_mode = self._learning_phase if hasattr(self, '_learning_phase') else False

        # Apply preprocessing and defences
        x_preproc = self._apply_processing(x)
        x_preproc, _ = self._apply_defences(x_preproc, None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preproc.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preproc) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preproc.shape[0])

            # Predict
            x_batch = mx.nd.array(x_preproc[begin:end], ctx=self._ctx)
            x_batch.attach_grad()
            with mx.autograd.record(train_mode=train_mode):
                preds = self._model(x_batch)

            if logits is False:
                preds = preds.softmax()

            results[begin:end] = preds.asnumpy()

        return results

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        import mxnet as mx

        # Check value of label for computing gradients
        if not (label is None or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
                or (isinstance(label, np.ndarray) and len(label.shape) == 1 and (label < self.nb_classes).all()
                    and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % str(label))

        train_mode = self._learning_phase if hasattr(self, '_learning_phase') else False

        x_preproc = self._apply_processing(x)
        x_defences, _ = self._apply_defences(x_preproc, None, fit=False)
        x_defences = mx.nd.array(x_defences, ctx=self._ctx)
        x_defences.attach_grad()

        if label is None:
            with mx.autograd.record(train_mode=False):
                if logits is True:
                    preds = self._model(x_defences)
                else:
                    preds = self._model(x_defences).softmax()
                class_slices = [preds[:, i] for i in range(self.nb_classes)]

            grads = []
            for slice_ in class_slices:
                slice_.backward(retain_graph=True)
                grad = x_defences.grad.asnumpy()
                grads.append(grad)
            grads = np.swapaxes(np.array(grads), 0, 1)
        elif isinstance(label, (int, np.integer)):
            with mx.autograd.record(train_mode=train_mode):
                if logits is True:
                    preds = self._model(x_defences)
                else:
                    preds = self._model(x_defences).softmax()
                class_slice = preds[:, label]

            class_slice.backward()
            grads = np.expand_dims(x_defences.grad.asnumpy(), axis=1)
        else:
            unique_labels = list(np.unique(label))

            with mx.autograd.record(train_mode=train_mode):
                if logits is True:
                    preds = self._model(x_defences)
                else:
                    preds = self._model(x_defences).softmax()
                class_slices = [preds[:, i] for i in unique_labels]

            grads = []
            for slice_ in class_slices:
                slice_.backward(retain_graph=True)
                grad = x_defences.grad.asnumpy()
                grads.append(grad)

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_labels.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]
            grads = np.expand_dims(grads, axis=1)

        grads = self._apply_defences_gradient(x_preproc, grads)
        grads = self._apply_processing_gradient(grads)

        return grads

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
        import mxnet as mx

        train_mode = self._learning_phase if hasattr(self, '_learning_phase') else False

        x_preproc = self._apply_processing(x)
        x_defences, y_defences = self._apply_defences(x_preproc, y, fit=False)
        y_defences = mx.nd.array([np.argmax(y_defences, axis=1)]).T
        x_defences = mx.nd.array(x_defences, ctx=self._ctx)
        x_defences.attach_grad()

        loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        with mx.autograd.record(train_mode=train_mode):
            preds = self._model(x_defences)
            loss = loss(preds, y_defences)

        loss.backward()
        grads = x_defences.grad.asnumpy()
        grads = self._apply_defences_gradient(x_preproc, grads)
        grads = self._apply_processing_gradient(grads)
        assert grads.shape == x.shape

        return grads

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
        return self._layer_names

    def get_activations(self, x, layer, batch_size=128):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        import mxnet as mx

        train_mode = self._learning_phase if hasattr(self, '_learning_phase') else False

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the model.' % layer)
            layer_ind = self._layer_names.index(layer)
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_ind = layer
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        # Apply preprocessing and defences
        if x.shape == self.input_shape:
            x_preproc = np.expand_dims(x, 0)
        else:
            x_preproc = x
        x_preproc = self._apply_processing(x_preproc)
        x_preproc, _ = self._apply_defences(x_preproc, None, fit=False)

        # Compute activations with batching
        activations = []
        nb_batches = int(np.ceil(len(x_preproc) / float(batch_size)))
        for batch_index in range(nb_batches):
            # Batch indexes
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preproc.shape[0])

            # Predict
            x_batch = mx.nd.array(x_preproc[begin:end], ctx=self._ctx)
            x_batch.attach_grad()
            with mx.autograd.record(train_mode=train_mode):
                preds = self._model[layer_ind](x_batch)

            activations.append(preds.asnumpy())

        activations = np.vstack(activations)
        return activations

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        if isinstance(train, bool):
            self._learning_phase = train

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For Gluon, only parameters are saved in
        file with name `<filename>.params` at the specified path. To load the saved model, the original model code needs
        to be run before calling `load_parameters` on the generated Gluon model.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os

        if path is None:
            from art import DATA_PATH
            full_path = os.path.join(DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        self._model.save_parameters(full_path + '.params')
        logger.info("Model parameters saved in path: %s.params.", full_path)

    def __repr__(self):
        repr_ = "%s(clip_values=%r, model=%r, input_shape=%r, nb_classes=%r, optimizer=%r, ctx=%r, channel_index=%r, " \
                "defences=%r, preprocessing=%r)" \
                % (self.__module__ + '.' + self.__class__.__name__,
                   self.clip_values, self._model, self.input_shape, self.nb_classes, self._optimizer, self._ctx,
                   self.channel_index, self.defences, self.preprocessing)

        return repr_

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        layer_names = [layer.name for layer in self._model[:-1]]
        logger.info('Inferred %i hidden layers on MXNet classifier.', len(layer_names))

        return layer_names
