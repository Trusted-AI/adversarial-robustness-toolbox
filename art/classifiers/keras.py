from __future__ import absolute_import, division, print_function, unicode_literals

import six
import numpy as np

from art.classifiers.classifier import Classifier, ImageClassifier, TextClassifier


class KerasClassifier(Classifier):
    """
    The supported backends for Keras are TensorFlow and Theano.
    """
    def __init__(self, model, loss, use_logits=False, input_layer=0, output_layer=0):
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param model: Keras model
        :type model: `keras.models.Model`
        :param loss: Loss function between true and predicted labels (encoded as one-hot)
        :type loss: `Callable`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param input_layer: Which layer to consider as the input when the model has multiple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the output when the model has multiple output layers.
        :type output_layer: `int`
        """
        import keras.backend as k

        super(KerasClassifier, self).__init__()

        self._model = model
        if hasattr(model, 'inputs'):
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input

        if hasattr(model, 'outputs'):
            self._output = model.outputs[output_layer]
        else:
            self._output = model.output

        _, self._nb_classes = k.int_shape(self._output)

        # Get predictions and loss function
        label_ph = k.placeholder(shape=self._output.shape)
        if not use_logits:
            if k.backend() == 'tensorflow':
                preds, = self._output.op.inputs
                loss_ = loss(label_ph, preds, from_logits=True)
            else:
                loss_ = loss(label_ph, self._output, from_logits=use_logits)

                # Convert predictions to logits for consistency with the other cases
                eps = 10e-8
                preds = k.log(k.clip(self._output, eps, 1. - eps))
        else:
            preds = self._output
            loss_ = loss(label_ph, self._output, from_logits=use_logits)
        loss_grads = k.gradients(loss_, self._input)

        if k.backend() == 'tensorflow':
            loss_grads = loss_grads[0]
        elif k.backend() == 'cntk':
            raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

        # Set loss, grads and prediction functions
        self._preds_op = preds
        self._loss_op = loss_
        self._loss = k.function([self._input], [loss_])
        self._preds = k.function([self._input], [preds])

        try:
            self._loss_grads = k.function([self._input, label_ph], [loss_grads])
        except TypeError:
            pass

        # Get the internal layer
        self._layer_names = self._get_layers()

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
        x_ = self._apply_processing(x)
        grads = self._loss_grads([x_, y])[0]
        grads = self._apply_processing_gradient(grads)
        assert grads.shape == x_.shape

        return grads

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        if label is not None and label not in range(self._nb_classes):
            raise ValueError('Label %s is out of range.' % label)

        self._init_class_grads(label=label, logits=logits)

        x_ = self._apply_processing(x)

        if label is not None:
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits_idx[label]([x_])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads_idx[label]([x_])), 0, 1)

            grads = self._apply_processing_gradient(grads)
            assert grads.shape == (x_.shape[0], 1) + self.input_shape
        else:
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits([x_])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads([x_])), 0, 1)

            grads = self._apply_processing_gradient(grads)
            assert grads.shape == (x_.shape[0], self.nb_classes) + self.input_shape

        return grads

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
        import keras.backend as k
        k.set_learning_phase(0)

        # Apply defences
        x_ = self._apply_processing(x)
        x_ = self._apply_defences_predict(x_)

        # Run predictions with batching
        preds = np.zeros((x_.shape[0], self.nb_classes), dtype=np.float32)
        for b in range(int(np.ceil(x_.shape[0] / float(batch_size)))):
            begin, end = b * batch_size,  min((b + 1) * batch_size, x_.shape[0])
            preds[begin:end] = self._preds([x_[begin:end]])[0]

            if not logits:
                exp = np.exp(preds[begin:end] - np.max(preds[begin:end], axis=1, keepdims=True))
                preds[begin:end] = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def fit(self, x, y, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        import keras.backend as k
        k.set_learning_phase(1)

        # Apply preprocessing and defences
        x_ = self._apply_processing(x)
        x_, y_ = self._apply_defences_fit(x_, y)

        gen = generator_fit(x_, y_, batch_size)
        self._model.fit_generator(gen, steps_per_epoch=x_.shape[0] / batch_size, epochs=nb_epochs)

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

    def get_activations(self, x, layer):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
        elif type(layer) is int:
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_name = self._layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        layer_output = self._model.get_layer(layer_name).output
        output_func = k.function([self._input], [layer_output])

        # Apply preprocessing and defences
        if x.shape == self.input_shape:
            x_ = np.expand_dims(x, 0)
        else:
            x_ = x
        x_ = self._apply_processing(x_)
        x_ = self._apply_defences_predict(x_)

        return output_func([x_])[0]

    def _init_class_grads(self, label=None, logits=False):
        import keras.backend as k
        k.set_learning_phase(0)

        if label is not None:
            if logits:
                if not hasattr(self, '_class_grads_logits_idx'):
                    self._class_grads_logits_idx = [None for _ in range(self.nb_classes)]

                if self._class_grads_logits_idx[label] is None:
                    class_grads_logits = [k.gradients(self._preds_op[:, label], self._input)[0]]
                    self._class_grads_logits_idx[label] = k.function([self._input], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads_idx'):
                    self._class_grads_idx = [None for _ in range(self.nb_classes)]

                if self._class_grads_idx[label] is None:
                    class_grads = [k.gradients(k.softmax(self._preds_op)[:, label], self._input)[0]]
                    self._class_grads_idx[label] = k.function([self._input], class_grads)
        else:
            if logits:
                if not hasattr(self, '_class_grads_logits'):
                    class_grads_logits = [k.gradients(self._preds_op[:, i], self._input)[0]
                                          for i in range(self.nb_classes)]
                    self._class_grads_logits = k.function([self._input], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads'):
                    class_grads = [k.gradients(k.softmax(self._preds_op)[:, i], self._input)[0]
                                   for i in range(self.nb_classes)]
                    self._class_grads = k.function([self._input], class_grads)

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        from keras.engine.topology import InputLayer

        layer_names = [layer.name for layer in self._model.layers[:-1] if not isinstance(layer, InputLayer)]
        return layer_names


class KerasImageClassifier(ImageClassifier, KerasClassifier):
    def __init__(self, clip_values, model, loss, use_logits=False, channel_index=3, defences=None, preprocessing=(0, 1),
                 input_layer=0, output_layer=0):
        """
        Create a :class:`KerasImageClassifier` instance from a Keras model. Assumes the `model` passed as argument is
        compiled.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: Keras model
        :type model: `keras.models.Model`
        :param loss: Loss function between true and predicted labels (encoded as one-hot)
        :type loss: `Callable`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param input_layer: Which layer to consider as the input when the model has multiple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the output when the model has multiple output layers.
        :type output_layer: `int`
        """
        import keras.backend as k

        ImageClassifier.__init__(self, clip_values=clip_values, channel_index=channel_index, defences=defences,
                                 preprocessing=preprocessing)

        KerasClassifier.__init__(self, model=model, loss=loss, use_logits=use_logits, input_layer=input_layer,
                                 output_layer=output_layer)

        self._input_shape = k.int_shape(self._input)[1:]
        self._channel_index = channel_index


class KerasTextClassifier(TextClassifier, KerasClassifier):
    """
    Class providing an implementation for integrating text models from Keras.
    """
    def __init__(self, model, loss, ids, use_logits=False, embedding_layer=0, input_layer=0, output_layer=0):
        """
        Create a :class:`KerasTextClassifier` instance from a Keras model. Assumes the `model` passed as argument is
        compiled.

        :param model: Keras model
        :type model: `keras.models.Model`
        :param loss: Loss function between true and predicted labels (encoded as one-hot)
        :type loss: `Callable`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param embedding_layer: Which layer to consider as providing the embedding of the vocabulary. When using the
               Keras `Embedding` layer, this can only be the first layer in the model.
        :type embedding_layer: `int`
        :param input_layer: Which layer to consider as the input when the model has multiple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the output when the model has multiple output layers.
        :type output_layer: `int`
        """
        import keras.backend as k

        TextClassifier.__init__(self)
        KerasClassifier.__init__(self, model=model, loss=loss, use_logits=use_logits, input_layer=input_layer,
                                 output_layer=output_layer)

        if type(embedding_layer) is int:
            embedding_name = self._layer_names[embedding_layer]
        else:
            raise ValueError('Expected `int` for `embedding_layer`, got %s.' % str(type(embedding_layer)))

        self._embedding = self._model.get_layer(embedding_name)
        self._embedding_from_input = k.function([self._input], [self._embedding.output])
        self._preds_from_embedding = k.function([self._embedding.output], [self._preds_op])
        self._ids = ids

        if not hasattr(self, '_loss_grads') or self._loss_grads is None:
            label_ph = k.placeholder(shape=self._output.shape)
            loss_grads = k.gradients(self._loss_op, self._embedding.output)

            if k.backend() == 'tensorflow':
                loss_grads = loss_grads[0]
            elif k.backend() == 'cntk':
                raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

            self._loss_grads = k.function([self._input, label_ph], [loss_grads])

    def predict_from_embedding(self, x_emb, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs in embedding form.

        :param x_emb: Array of inputs in embedding form, often shaped as `(batch_size, input_length, embedding_size)`.
        :type x_emb: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        # Run predictions with batching
        preds = np.zeros((x_emb.shape[0], self.nb_classes), dtype=np.float32)
        for b in range(int(np.ceil(x_emb.shape[0] / float(batch_size)))):
            begin, end = b * batch_size,  min((b + 1) * batch_size, x_emb.shape[0])
            preds[begin:end] = self._preds_from_embedding([x_emb[begin:end]])[0]

            if not logits:
                exp = np.exp(preds[begin:end] - np.max(preds[begin:end], axis=1, keepdims=True))
                preds[begin:end] = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def to_embedding(self, x):
        """
        Convert the received classifier input `x` from token (words or characters) indices to embeddings.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Embedding form of sample `x`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        return self._embedding_from_input([x])

    def to_id(self, x_emb, strategy='nearest', metric='cosine'):
        """
        Convert the received input from embedding space to classifier input (most often, token indices).

        :param x_emb: Array of inputs in embedding form, often shaped as `(batch_size, input_length, embedding_size)`.
        :type x_emb: `np.ndarray`
        :param strategy: Strategy from mapping from embedding space back to input space.
        :type strategy: `str` or `Callable`
        :param metric: Metric to be used in the embedding space when determining vocabulary token proximity.
        :type metric: `str` or `Callable`
        :return: Array of token indices for sample `x_emb`.
        :rtype: `np.ndarray`
        """
        import keras.backend as k
        k.set_learning_phase(0)

        if strategy != 'nearest':
            raise ValueError('Nearest neighbor is currently the only supported strategy for mapping embeddings to '
                             'valid tokens.')

        if metric == 'cosine':
            from art.utils import cosine

            embeddings = self.to_embedding(self._ids)

            neighbors = []
            for x in x_emb:
                metric = [cosine(emb, x) for emb in embeddings]
                neighbors.append(self._ids[int(np.argpartition(metric, -1)[-1])])
        else:
            raise ValueError('Cosine similarity is currently the only supported metric for mapping embeddings to '
                             'valid tokens.')

        return np.array(neighbors)


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
