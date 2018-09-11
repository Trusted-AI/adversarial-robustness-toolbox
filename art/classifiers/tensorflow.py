from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random
import six

from art.classifiers.classifier import Classifier, ImageClassifier, TextClassifier


class TFClassifier(Classifier):
    """
    This class implements a classifier with the Tensorflow framework.
    """
    def __init__(self, input_ph, logits, output_ph=None, train=None, loss=None, learning=None, sess=None):
        """
        Initialization specifically for the Tensorflow-based implementation.

        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param logits: The logits layer of the model.
        :type logits: `tf.Tensor`
        :param output_ph: The labels placeholder of the model. This parameter is necessary when training the model and
               when computing gradients w.r.t. the loss function.
        :type output_ph: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
               model and when computing gradients w.r.t. the loss function.
        :type loss: `tf.Tensor`
        :param learning: The placeholder to indicate if the model is training.
        :type learning: `tf.Placeholder` of type bool.
        :param sess: Computation session.
        :type sess: `tf.Session`
        """
        import tensorflow as tf

        super(TFClassifier, self).__init__()
        self._nb_classes = int(logits.get_shape()[-1])
        self._input_ph = input_ph
        self._logits = logits
        self._output_ph = output_ph
        self._train = train
        self._loss = loss
        self._learning = learning
        self._input_shape = tuple(input_ph.get_shape()[1:])

        # Assign session
        if sess is None:
            # self._sess = tf.get_default_session()
            raise ValueError("A session cannot be None.")
        else:
            self._sess = sess

        # Get the internal layers
        self._layer_names = self._get_layers()

        # Must be set here for the softmax output
        self._probs = tf.nn.softmax(logits)

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

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
        # Apply defences
        x_ = self._apply_processing(x)
        x_ = self._apply_defences_predict(x_)

        # Run prediction with batch processing
        results = np.zeros((x_.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_.shape[0])

            # Create feed_dict
            fd = {self._input_ph: x_[begin:end]}
            if self._learning is not None:
                fd[self._learning] = False

            # Run prediction
            if logits:
                results[begin:end] = self._sess.run(self._logits, feed_dict=fd)
            else:
                results[begin:end] = self._sess.run(self._probs, feed_dict=fd)

        return results

    def fit(self, x, y, batch_size=128, nb_epochs=10):
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
        # Check if train and output_ph available
        if self._train is None or self._output_ph is None:
            raise ValueError("Need the training objective and the output placeholder to train the model.")

        # Apply defences
        x_ = self._apply_processing(x)
        x_, y_ = self._apply_defences_fit(x_, y)

        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        ind = np.arange(len(x_))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = x_[ind[m * batch_size:(m + 1) * batch_size]]
                o_batch = y_[ind[m * batch_size:(m + 1) * batch_size]]

                # Run train step
                if self._learning is None:
                    self._sess.run(self._train, feed_dict={self._input_ph: i_batch, self._output_ph: o_batch})
                else:
                    self._sess.run(self._train, feed_dict={self._input_ph: i_batch, self._output_ph: o_batch,
                                                           self._learning: True})

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int` or `numpy.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        if not ((label is None) or (type(label) is int and label in range(self._nb_classes)) or (
            type(label) is np.ndarray and len(label.shape) == 1 and (label < self._nb_classes).all()
            and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % label)

        self._init_class_grads(label=label, logits=logits)

        x_ = self._apply_processing(x)

        # Compute the gradient and return
        if label is None:
            if logits:
                grads = self._sess.run(self._logit_class_grads, feed_dict={self._input_ph: x_})
            else:
                grads = self._sess.run(self._class_grads, feed_dict={self._input_ph: x_})

            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)
            assert grads.shape == (x_.shape[0], self.nb_classes) + self.input_shape

        elif type(label) is int:
            if logits:
                grads = self._sess.run(self._logit_class_grads[label], feed_dict={self._input_ph: x_})
            else:
                grads = self._sess.run(self._class_grads[label], feed_dict={self._input_ph: x_})

            grads = grads[None, ...]
            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)
            assert grads.shape == (x_.shape[0], 1) + self.input_shape

        else:
            unique_label = list(np.unique(label))
            if logits:
                grads = self._sess.run([self._logit_class_grads[l] for l in unique_label],
                                       feed_dict={self._input_ph: x_})
            else:
                grads = self._sess.run([self._class_grads[l] for l in unique_label],
                                       feed_dict={self._input_ph: x_})

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]
            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)
            assert grads.shape == (x_.shape[0], 1) + self.input_shape

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
        x_ = self._apply_processing(x)

        # Check if loss available
        if not hasattr(self, '_loss_grads') or self._loss_grads is None or self._output_ph is None:
            raise ValueError("Need the loss function and the labels placeholder to compute the loss gradient.")

        # Compute the gradient and return
        grds = self._sess.run(self._loss_grads, feed_dict={self._input_ph: x_, self._output_ph: y})
        grds = self._apply_processing_gradient(grds)
        assert grds.shape == x_.shape

        return grds

    def _gen_init_class_grads(self, input_tensor, label=None, logits=False):
        """
        Add more operations to the tensorflow graph to compute class gradients.

        :param input_tensor: Compute class gradient wrt input_tensor.
        :type input_tensor: `tf.Tensor`
        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int` or `numpy.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: `None`
        """
        import tensorflow as tf

        if logits:
            if not hasattr(self, '_logit_class_grads'):
                self._logit_class_grads = [None for _ in range(self.nb_classes)]
        else:
            if not hasattr(self, '_class_grads'):
                self._class_grads = [None for _ in range(self.nb_classes)]

        # Construct the class gradients graph
        if label is None:
            if logits:
                if None in self._logit_class_grads:
                    self._logit_class_grads = [tf.gradients(self._logits[:, i], input_tensor)[0]
                                               if self._logit_class_grads[i] is None else self._logit_class_grads[i]
                                               for i in range(self._nb_classes)]
            else:
                if None in self._class_grads:
                    self._class_grads = [tf.gradients(self._probs[:, i], input_tensor)[0]
                                         if self._class_grads[i] is None else self._class_grads[i]
                                         for i in range(self._nb_classes)]

        elif type(label) is int:
            if logits:
                if self._logit_class_grads[label] is None:
                    self._logit_class_grads[label] = tf.gradients(self._logits[:, label], input_tensor)[0]
            else:
                if self._class_grads[label] is None:
                    self._class_grads[label] = tf.gradients(self._probs[:, label], input_tensor)[0]

        else:
            if logits:
                for l in np.unique(label):
                    if self._logit_class_grads[l] is None:
                        self._logit_class_grads[l] = tf.gradients(self._logits[:, l], input_tensor)[0]
            else:
                for l in np.unique(label):
                    if self._class_grads[l] is None:
                        self._class_grads[l] = tf.gradients(self._probs[:, l], input_tensor)[0]

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
        import tensorflow as tf

        # Get the computational graph
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        if isinstance(layer, six.string_types):  # basestring for Python 2 (str, unicode) support
            if layer not in self._layer_names:
                raise ValueError("Layer name %s is not part of the graph." % layer)
            layer_tensor = graph.get_tensor_by_name(layer)

        elif isinstance(layer, (int, np.integer)):
            layer_tensor = graph.get_tensor_by_name(self._layer_names[layer])

        else:
            raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)

        # Get activations
        # Apply preprocessing and defences
        x_ = self._apply_processing(x)
        x_ = self._apply_defences_predict(x_)

        # Create feed_dict
        fd = {self._input_ph: x_}
        if self._learning is not None:
            fd[self._learning] = False

        # Run prediction
        result = self._sess.run(layer_tensor, feed_dict=fd)

        return result


class TFImageClassifier(ImageClassifier, TFClassifier):
    def __init__(self, clip_values, input_ph, logits, output_ph=None, train=None, loss=None, learning=None, sess=None,
                 channel_index=3, defences=None, preprocessing=(0, 1)):
        """
        Initialize a :class:`TFImageClassifier` based on a Tensorflow image model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param logits: The logits layer of the model.
        :type logits: `tf.Tensor`
        :param output_ph: The labels placeholder of the model. This parameter is necessary when training the model and
               when computing gradients w.r.t. the loss function.
        :type output_ph: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
               model and when computing gradients w.r.t. the loss function.
        :type loss: `tf.Tensor`
        :param learning: The placeholder to indicate if the model is training.
        :type learning: `tf.Placeholder` of type bool.
        :param sess: Computation session.
        :type sess: `tf.Session`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        ImageClassifier.__init__(self, clip_values=clip_values, channel_index=channel_index, defences=defences,
                                 preprocessing=preprocessing)

        TFClassifier.__init__(self, input_ph=input_ph, logits=logits, output_ph=output_ph, train=train, loss=loss,
                              learning=learning, sess=sess)

    def _init_class_grads(self, label=None, logits=False):
        """
        Add more operations to the tensorflow graph to compute class gradients.

        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int` or `numpy.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: `None`
        """
        self._gen_init_class_grads(input_tensor=self._input_ph, label=label, logits=logits)

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        import tensorflow as tf

        # Get the computational graph
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        # Get the list of operators and heuristically filter them
        tmp_list = []
        ops = graph.get_operations()

        for op in ops:
            filter_cond = ((op.values()) and (not op.values()[0].get_shape() == None) and (
                len(op.values()[0].get_shape().as_list()) > 1) and (
                op.values()[0].get_shape().as_list()[0] is None) and (
                op.values()[0].get_shape().as_list()[1] is not None) and (
                not op.values()[0].name.startswith("gradients")) and (
                not op.values()[0].name.startswith("softmax_cross_entropy_loss")) and (
                not op.type == "Placeholder"))

            if filter_cond:
                tmp_list.append(op.values()[0].name)

        # Shorten the list
        if len(tmp_list) == 0:
            return tmp_list

        result = [tmp_list[-1]]
        for name in reversed(tmp_list[:-1]):
            if result[0].split("/")[0] != name.split("/")[0]:
                result = [name] + result

        return result


class TFTextClassifier(TextClassifier, TFClassifier):
    def __init__(self, input_ph, logits, embedding_layer, ids, output_ph=None, train=None, loss=None, learning=None,
                 sess=None):
        """
        Create a :class:`TFTextClassifier` instance from a Tensorflow model.

        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param logits: The logits layer of the model.
        :type logits: `tf.Tensor`
        :param embedding_layer: Which layer to consider as providing the embedding of the vocabulary.
        :type embedding_layer: `tf.Tensor`
        :param ids: List of ids in the vocabulary.
        :type ids: `list`
        :param output_ph: The labels placeholder of the model. This parameter is necessary when training the model and
               when computing gradients w.r.t. the loss function.
        :type output_ph: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
               model and when computing gradients w.r.t. the loss function.
        :type loss: `tf.Tensor`
        :param learning: The placeholder to indicate if the model is training.
        :type learning: `tf.Placeholder` of type bool.
        :param sess: Computation session.
        :type sess: `tf.Session`
        """
        import tensorflow as tf

        TextClassifier.__init__(self)

        TFClassifier.__init__(self, input_ph=input_ph, logits=logits, output_ph=output_ph, train=train, loss=loss,
                              learning=learning, sess=sess)

        self._embedding_layer = embedding_layer
        self._ids = ids

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._embedding_layer)[0]

        # Define function for embedding prediction
        self._preds_from_embedding = tf.keras.backend.function([self._embedding_layer], [self._logits])

        # Define function for embedding convertion
        self._embedding_from_input = tf.keras.backend.function([self._input_ph], [self._embedding_layer])

    def _init_class_grads(self, label=None, logits=False):
        """
        Add more operations to the tensorflow graph to compute class gradients.

        :param label: Index of a specific per-class derivative. If `None`, then gradients for all
                      classes will be computed.
        :type label: `int` or `numpy.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: `None`
        """
        self._gen_init_class_grads(input_tensor=self._embedding_layer, label=label, logits=logits)

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        import tensorflow as tf

        # Get the computational graph
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        # Get the list of operators and heuristically filter them
        tmp_list = []
        ops = graph.get_operations()

        for op in ops:
            filter_cond = ((op.values()) and (not op.values()[0].get_shape() == None) and (
                len(op.values()[0].get_shape().as_list()) > 1) and (
                op.values()[0].get_shape().as_list()[0] is None) and (
                op.values()[0].get_shape().as_list()[1] is not None) and (
                not op.values()[0].name.startswith("gradients")) and (
                not op.values()[0].name.startswith("softmax_cross_entropy_loss")) and (
                not op.type == "Placeholder") and (len(op.values()[0].name.split("/")) < 4))

            if filter_cond:
                tmp_list.append(op.values()[0].name)

        # Shorten the list
        if len(tmp_list) == 0:
            return tmp_list

        result = [tmp_list[-1]]
        for name in reversed(tmp_list[:-1]):
            if result[0].split("/")[0] != name.split("/")[0]:
                result = [name] + result

        return result

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
        import tensorflow as tf
        tf.keras.backend.set_learning_phase(0)

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
        import tensorflow as tf
        tf.keras.backend.set_learning_phase(0)

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
        if strategy != 'nearest':
            raise ValueError('Nearest neighbor is currently the only supported strategy for mapping embeddings to '
                             'valid tokens.')

        if metric == 'cosine':
            from art.utils import cosine

            v_size = len(self._ids)
            if v_size % x_emb.shape[1] > 0:
                for _ in range(x_emb.shape[1] - (v_size % x_emb.shape[1])):
                    self._ids.append(self._ids[0])
            embeddings = self.to_embedding(np.reshape(np.array(self._ids), (-1, x_emb.shape[1])))
            embeddings = np.reshape(embeddings, (-1, x_emb.shape[2]))[:v_size]
            self._ids = self._ids[:v_size]

            neighbors = []
            for x in x_emb:
                for emb_x in x:
                    metric = [cosine(emb, emb_x) for emb in embeddings]
                    neighbors.append(self._ids[int(np.argpartition(metric, -1)[-1])])
        else:
            raise ValueError('Cosine similarity is currently the only supported metric for mapping embeddings to '
                             'valid tokens.')

        return np.reshape(np.array(neighbors), (-1, x_emb.shape[1]))















