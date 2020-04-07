# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the classifier `TensorFlowClassifier` for TensorFlow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random

import numpy as np
import six

from art.estimators.tensorflow import TensorFlowEstimator, TensorFlowV2Estimator
from art.estimators.classification.classifier import ClassifierMixin, ClassGradientsMixin

logger = logging.getLogger(__name__)


class TensorFlowClassifier(ClassGradientsMixin, ClassifierMixin, TensorFlowEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements a classifier with the TensorFlow framework.
    """

    def __init__(
        self,
        input_ph,
        output,
        labels_ph=None,
        train=None,
        loss=None,
        learning=None,
        sess=None,
        channel_index=3,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1),
        feed_dict={},
    ):
        """
        Initialization specific to TensorFlow models implementation.

        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param output: The output layer of the model. This can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :type output: `tf.Tensor`
        :param labels_ph: The labels placeholder of the model. This parameter is necessary when training the model and
               when computing gradients w.r.t. the loss function.
        :type labels_ph: `tf.Tensor`
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
        # pylint: disable=E0401
        import tensorflow as tf

        super(TensorFlowClassifier, self).__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._nb_classes = int(output.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        self._input_ph = input_ph
        self._output = output
        self._labels_ph = labels_ph
        self._train = train
        self._loss = loss
        self._learning = learning
        self._feed_dict = feed_dict

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
        self._sess = sess

        # Get the internal layers
        self._layer_names = self._get_layers()

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        if len(self._labels_ph.shape) == 1:
            self._reduce_labels = True
        else:
            self._reduce_labels = False

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(num_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

            # Create feed_dict
            feed_dict = {self._input_ph: x_preprocessed[begin:end]}
            feed_dict.update(self._feed_dict)

            # Run prediction
            results[begin:end] = self._sess.run(self._output, feed_dict=feed_dict)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for
               TensorFlow and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        # Check if train and output_ph available
        if self._train is None or self._labels_ph is None:
            raise ValueError("Need the training objective and the output placeholder to train the model.")

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]
                o_batch = y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]

                # Create feed_dict
                feed_dict = {self._input_ph: i_batch, self._labels_ph: o_batch}
                feed_dict.update(self._feed_dict)

                # Run train step
                self._sess.run(self._train, feed_dict=feed_dict)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in TensorFlow, it will.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for
               TensorFlow and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        from art.data_generators import TensorFlowDataGenerator

        # Train directly in TensorFlow
        if (
            isinstance(generator, TensorFlowDataGenerator)
            and (self.preprocessing_defences is None or self.preprocessing_defences == [])
            and self.preprocessing == (0, 1)
        ):
            for _ in range(nb_epochs):
                for _ in range(int(generator.size / generator.batch_size)):
                    i_batch, o_batch = generator.get_batch()

                    if self._reduce_labels:
                        o_batch = np.argmax(o_batch, axis=1)

                    # Create feed_dict
                    feed_dict = {self._input_ph: i_batch, self._labels_ph: o_batch}
                    feed_dict.update(self._feed_dict)

                    # Run train step
                    self._sess.run(self._train, feed_dict=feed_dict)
        else:
            super(TensorFlowClassifier, self).fit_generator(generator, nb_epochs=nb_epochs, **kwargs)

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        # Check value of label for computing gradients
        if not (
            label is None
            or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self.nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError("Label %s is out of range." % label)

        self._init_class_grads(label=label)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Create feed_dict
        feed_dict = {self._input_ph: x_preprocessed}
        feed_dict.update(self._feed_dict)

        # Compute the gradient and return
        if label is None:
            # Compute the gradients w.r.t. all classes
            grads = self._sess.run(self._class_grads, feed_dict=feed_dict)
            grads = np.swapaxes(np.array(grads), 0, 1)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            grads = self._sess.run(self._class_grads[label], feed_dict=feed_dict)
            grads = grads[None, ...]
            grads = np.swapaxes(np.array(grads), 0, 1)

        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            grads = self._sess.run([self._class_grads[l] for l in unique_label], feed_dict=feed_dict)
            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = np.expand_dims(grads[np.arange(len(grads)), lst], axis=1)

        grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check if loss available
        if not hasattr(self, "_loss_grads") or self._loss_grads is None or self._labels_ph is None:
            raise ValueError("Need the loss function and the labels placeholder to compute the loss gradient.")

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Create feed_dict
        feed_dict = {self._input_ph: x_preprocessed, self._labels_ph: y_preprocessed}
        feed_dict.update(self._feed_dict)

        # Compute gradients
        grads = self._sess.run(self._loss_grads, feed_dict=feed_dict)
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x_preprocessed.shape

        return grads

    def _init_class_grads(self, label=None):
        # pylint: disable=E0401
        import tensorflow as tf

        if not hasattr(self, "_class_grads"):
            self._class_grads = [None for _ in range(self.nb_classes)]

        # Construct the class gradients graph
        if label is None:
            if None in self._class_grads:
                self._class_grads = [
                    tf.gradients(self._output[:, i], self._input_ph)[0] for i in range(self.nb_classes)
                ]

        elif isinstance(label, int):
            if self._class_grads[label] is None:
                self._class_grads[label] = tf.gradients(self._output[:, label], self._input_ph)[0]

        else:
            for unique_label in np.unique(label):
                if self._class_grads[unique_label] is None:
                    self._class_grads[unique_label] = tf.gradients(self._output[:, unique_label], self._input_ph)[0]

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        # pylint: disable=E0401
        import tensorflow as tf

        # Get the computational graph
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        # Get the list of operators and heuristically filter them
        tmp_list = []
        ops = graph.get_operations()

        # pylint: disable=R1702
        for op in ops:
            if op.values():
                if op.values()[0].get_shape() is not None:
                    if op.values()[0].get_shape().ndims is not None:
                        if len(op.values()[0].get_shape().as_list()) > 1:
                            if op.values()[0].get_shape().as_list()[0] is None:
                                if op.values()[0].get_shape().as_list()[1] is not None:
                                    if not op.values()[0].name.startswith("gradients"):
                                        if not op.values()[0].name.startswith("softmax_cross_entropy_loss"):
                                            if not op.type == "Placeholder":
                                                tmp_list.append(op.values()[0].name)

        # Shorten the list
        if not tmp_list:
            return tmp_list

        result = [tmp_list[-1]]
        for name in reversed(tmp_list[:-1]):
            if result[0].split("/")[0] != name.split("/")[0]:
                result = [name] + result
        logger.info("Inferred %i hidden layers on TensorFlow classifier.", len(result))

        return result

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

    def get_activations(self, x, layer, batch_size=128, intermediate=False):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations.
        :type layer: `int` or `str`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0401
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
            raise TypeError("Layer must be of type `str` or `int`. Received %s." % layer)

        if intermediate:
            return layer_tensor

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = []
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

            # Create feed_dict
            feed_dict = {self._input_ph: x_preprocessed[begin:end]}
            feed_dict.update(self._feed_dict)

            # Run prediction for the current batch
            layer_output = self._sess.run(layer_tensor, feed_dict=feed_dict)
            results.append(layer_output)

        results = np.concatenate(results)

        return results

    def custom_gradient(self, nn_function):
        """
        Returns the gradient of the nn_function with respect to vars

        :param nn_function: an intermediate tensor representation of the gradient function
        :type nn_function: a Keras tensor
        :param vars: the variables to differentiate
        :type vars: `list`
        :return: the gradient of the function w.r.t vars
        :rtype: `np.ndarray`
        """
    def get_input_layer(self):
        return self._input

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        if isinstance(train, bool):
            self._learning_phase = train
            self._feed_dict[self._learning] = train

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For TensorFlow, .ckpt is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :type path: `str`
        """
        # pylint: disable=E0611
        import os
        import shutil
        from tensorflow.python import saved_model
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

        if path is None:
            from art.config import ART_DATA_PATH

            full_path = os.path.join(ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)

        if os.path.exists(full_path):
            shutil.rmtree(full_path)

        builder = saved_model.builder.SavedModelBuilder(full_path)
        signature = predict_signature_def(
            inputs={"SavedInputPhD": self._input_ph}, outputs={"SavedOutput": self._output}
        )
        builder.add_meta_graph_and_variables(
            sess=self._sess, tags=[tag_constants.SERVING], signature_def_map={"predict": signature}
        )
        builder.save()

        logger.info("Model saved in path: %s.", full_path)

    def __getstate__(self):
        """
        Use to ensure `TensorFlowClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """
        import time

        state = self.__dict__.copy()

        # Remove the unpicklable entries
        del state["_sess"]
        del state["_input_ph"]
        state["_output"] = self._output.name

        if self._labels_ph is not None:
            state["_labels_ph"] = self._labels_ph.name

        if self._loss is not None:
            state["_loss"] = self._loss.name

        if hasattr(self, "_loss_grads"):
            state["_loss_grads"] = self._loss_grads.name
        else:
            state["_loss_grads"] = False

        if self._learning is not None:
            state["_learning"] = self._learning.name

        if self._train is not None:
            state["_train"] = self._train.name

        if hasattr(self, "_class_grads"):
            state["_class_grads"] = [ts if ts is None else ts.name for ts in self._class_grads]
        else:
            state["_class_grads"] = False

        model_name = str(time.time())
        state["model_name"] = model_name
        self.save(model_name)

        return state

    def __setstate__(self, state):
        """
        Use to ensure `TensorFlowClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to TensorFlow
        # pylint: disable=E0611, E0401
        import os
        import tensorflow as tf
        from tensorflow.python.saved_model import tag_constants
        from art.config import ART_DATA_PATH

        full_path = os.path.join(ART_DATA_PATH, state["model_name"])

        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        loaded = tf.saved_model.loader.load(sess, [tag_constants.SERVING], full_path)

        # Recover session
        self._sess = sess

        # Recover input_ph
        input_tensor_name = loaded.signature_def["predict"].inputs["SavedInputPhD"].name
        self._input_ph = graph.get_tensor_by_name(input_tensor_name)

        # Recover output layer
        self._output = graph.get_tensor_by_name(state["_output"])

        # Recover labels' placeholder if any
        if state["_labels_ph"] is not None:
            self._labels_ph = graph.get_tensor_by_name(state["_labels_ph"])

        # Recover loss if any
        if state["_loss"] is not None:
            self._loss = graph.get_tensor_by_name(state["_loss"])

        # Recover loss_grads if any
        if state["_loss_grads"]:
            self._loss_grads = graph.get_tensor_by_name(state["_loss_grads"])
        else:
            self.__dict__.pop("_loss_grads", None)

        # Recover learning if any
        if state["_learning"] is not None:
            self._learning = graph.get_tensor_by_name(state["_learning"])

        # Recover train if any
        if state["_train"] is not None:
            self._train = graph.get_operation_by_name(state["_train"])

        # Recover class_grads if any
        if state["_class_grads"]:
            self._class_grads = [ts if ts is None else graph.get_tensor_by_name(ts) for ts in state["_class_grads"]]
        else:
            self.__dict__.pop("_class_grads", None)

        self.__dict__.pop("model_name", None)

    def __repr__(self):
        repr_ = (
            "%s(input_ph=%r, output=%r, labels_ph=%r, train=%r, loss=%r, learning=%r, sess=%r, channel_index=%r, "
            "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._input_ph,
                self._output,
                self._labels_ph,
                self._train,
                self._loss,
                self._learning,
                self._sess,
                self.channel_index,
                self.clip_values,
                self.preprocessing_defences,
                self.postprocessing_defences,
                self.preprocessing,
            )
        )

        return repr_


# backward compatibility for ART v0.10 and earlier
TFClassifier = TensorFlowClassifier


class TensorFlowV2Classifier(ClassGradientsMixin, ClassifierMixin, TensorFlowV2Estimator):
    """
    This class implements a classifier with the TensorFlow v2 framework.
    """

    def __init__(
        self,
        model,
        nb_classes,
        input_shape,
        loss_object=None,
        train_step=None,
        channel_index=3,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1),
    ):
        """
        Initialization specific to TensorFlow v2 models.

        :param model: a python functions or callable class defining the model and providing it prediction as output.
        :type model: `function` or `callable class`
        :param nb_classes: the number of classes in the classification task.
        :type nb_classes: `int`
        :param input_shape: shape of one input for the classifier, e.g. for MNIST input_shape=(28, 28, 1).
        :type input_shape: `tuple`
        :param loss_object: The loss function for which to compute gradients. This parameter is applied for training
            the model and computing gradients of the loss w.r.t. the input.
        :type loss_object: `tf.keras.losses`
        :param train_step: a function that applies a gradient update to the trainable variables.
        :type train_step: `function`
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
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        import tensorflow as tf

        super(TensorFlowV2Classifier, self).__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._model = model
        self._nb_classes = nb_classes
        self._input_shape = input_shape
        self._loss_object = loss_object
        self._train_step = train_step

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        if isinstance(self._loss_object, tf.keras.losses.SparseCategoricalCrossentropy):
            self._reduce_labels = True
        else:
            self._reduce_labels = False

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_preprocessed.shape[0])

            # Run prediction
            results[begin:end] = self._model(x_preprocessed[begin:end])

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions

    def _predict_framework(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        return self._model(x_preprocessed)

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for
               TensorFlow and providing it takes no effect.
        :type kwargs: `dict`
        """
        import tensorflow as tf

        if self._train_step is None:
            raise TypeError(
                "The training function `train_step` is required for fitting a model but it has not been " "defined."
            )

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        train_ds = tf.data.Dataset.from_tensor_slices((x_preprocessed, y_preprocessed)).shuffle(10000).batch(batch_size)

        for _ in range(nb_epochs):
            for images, labels in train_ds:
                self._train_step(images, labels)

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch. If the generator can be used for native
                          training in TensorFlow, it will.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for
               TensorFlow and providing it takes no effect.
        :type kwargs: `dict`
        """
        import tensorflow as tf
        from art.data_generators import TensorFlowV2DataGenerator

        # Train directly in TensorFlow
        if (
            isinstance(generator, TensorFlowV2DataGenerator)
            and (self.preprocessing_defences is None or self.preprocessing_defences == [])
            and self.preprocessing == (0, 1)
        ):
            for _ in range(nb_epochs):
                for i_batch, o_batch in generator.iterator:
                    if self._reduce_labels:
                        o_batch = tf.math.argmax(o_batch, axis=1)
                    self._train_step(i_batch, o_batch)
        else:
            # Fit a generic data generator through the API
            super().fit_generator(generator, nb_epochs=nb_epochs)

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        import tensorflow as tf

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Compute the gradients
        if tf.executing_eagerly():
            if label is None:
                # Compute the gradients w.r.t. all classes
                class_gradients = list()

                for i in range(self.nb_classes):
                    with tf.GradientTape() as tape:
                        x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)
                        tape.watch(x_preprocessed_tf)
                        predictions = self._model(x_preprocessed_tf)
                        prediction = predictions[:, i]
                        tape.watch(prediction)

                    class_gradient = tape.gradient(prediction, x_preprocessed_tf).numpy()
                    class_gradients.append(class_gradient)

                gradients = np.swapaxes(np.array(class_gradients), 0, 1)

            elif isinstance(label, (int, np.integer)):
                # Compute the gradients only w.r.t. the provided label
                with tf.GradientTape() as tape:
                    x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)
                    tape.watch(x_preprocessed_tf)
                    predictions = self._model(x_preprocessed_tf)
                    prediction = predictions[:, label]
                    tape.watch(prediction)

                class_gradient = tape.gradient(prediction, x_preprocessed_tf).numpy()
                gradients = np.expand_dims(class_gradient, axis=1)

            else:
                # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
                class_gradients = list()
                unique_labels = list(np.unique(label))

                for unique_label in unique_labels:
                    with tf.GradientTape() as tape:
                        x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)
                        tape.watch(x_preprocessed_tf)
                        predictions = self._model(x_preprocessed_tf)
                        prediction = predictions[:, unique_label]
                        tape.watch(prediction)

                    class_gradient = tape.gradient(prediction, x_preprocessed_tf).numpy()
                    class_gradients.append(class_gradient)

                gradients = np.swapaxes(np.array(class_gradients), 0, 1)
                lst = [unique_labels.index(i) for i in label]
                gradients = np.expand_dims(gradients[np.arange(len(gradients)), lst], axis=1)

        else:
            raise ValueError("Expecting eager execution.")

        return gradients

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import tensorflow as tf

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y, fit=False)

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)
                tape.watch(x_preprocessed_tf)
                predictions = self._model(x_preprocessed_tf)
                if self._reduce_labels:
                    loss = self._loss_object(np.argmax(y, axis=1), predictions)
                else:
                    loss = self._loss_object(y, predictions)

            gradients = tape.gradient(loss, x_preprocessed_tf).numpy()
        else:
            raise ValueError("Expecting eager execution.")

        # Apply preprocessing gradients
        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """
        raise NotImplementedError

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
        import tensorflow as tf
        if isinstance(self._model, tf.keras.Model) or isinstance(self._model, tf.keras.model.Sequential):
            return self._model.layers
        else:
            return None

    def get_activations(self, x, layer, batch_size=128):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations.
        :type layer: `int` or `str`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        import tensorflow as tf
        from art.config import ART_NUMPY_DTYPE

        if isinstance(self._model, tf.keras.models.Sequential):
            i_layer = None
            if isinstance(layer, six.string_types):
                if layer not in self.layer_names:
                    raise ValueError("Layer name %s is not part of the graph." % layer)
                for i_name, name in enumerate(self.layer_names):
                    if name == layer:
                        i_layer = i_name
                        break
            elif isinstance(layer, int):
                if layer < 0 or layer >= len(self.layer_names):
                    raise ValueError(
                        "Layer index %d is outside of range (0 to %d included)." % (layer, len(self.layer_names) - 1)
                    )
                i_layer = layer
            else:
                raise TypeError("Layer must be of type `str` or `int`.")

            activation_model = tf.keras.Model(self._model.layers[0].input, self._model.layers[i_layer].output)

            # Apply preprocessing
            x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False)

            # Determine shape of expected output and prepare array
            output_shape = self._model.layers[i_layer].output_shape
            activations = np.zeros((x_preprocessed.shape[0],) + output_shape[1:], dtype=ART_NUMPY_DTYPE)

            # Get activations with batching
            for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
                begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preprocessed.shape[0])
                activations[begin:end] = activation_model([x_preprocessed[begin:end]]).numpy()

            return activations
        else:
            return None

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: True to set the learning phase to training, False to set it to prediction.
        :type train: `bool`
        """
        raise NotImplementedError

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For TensorFlow, .ckpt is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :type path: `str`
        """
        raise NotImplementedError

    def __repr__(self):
        repr_ = (
            "%s(model=%r, nb_classes=%r, input_shape=%r, loss_object=%r, train_step=%r, channel_index=%r, "
            "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r)"
            % (
                self.__module__ + "." + self.__class__.__name__,
                self._model,
                self._nb_classes,
                self._input_shape,
                self._loss_object,
                self._train_step,
                self.channel_index,
                self.clip_values,
                self.preprocessing_defences,
                self.postprocessing_defences,
                self.preprocessing,
            )
        )

        return repr_
