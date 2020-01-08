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
"""
Module providing convenience functions specifically for unit tests.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import json
import numpy as np

logger = logging.getLogger(__name__)

try:
    # Conditional import of `torch` to avoid segmentation fault errors this framework generates at import
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    logger.info('Could not import PyTorch in utilities.')


# ----------------------------------------------------------------------------------------------- TEST MODELS FOR MNIST


def _tf_weights_loader(dataset, weights_type, layer='DENSE', tf_version=1):
    filename = str(weights_type) + '_' + str(layer) + '_' + str(dataset) + '.npy'

    # pylint: disable=W0613
    # disable pylint because of API requirements for function
    if tf_version == 1:
        def _tf_initializer(_, dtype, partition_info):
            import tensorflow as tf

            weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', filename))
            return tf.constant(weights, dtype)

    elif tf_version == 2:
        def _tf_initializer(_, dtype):
            import tensorflow as tf

            weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', filename))
            return tf.constant(weights, dtype)

    else:
        raise ValueError('The TensorFlow version tf_version has to be wither 1 or 2.')

    return _tf_initializer


def _kr_weights_loader(dataset, weights_type, layer='DENSE'):
    import keras.backend as k
    filename = str(weights_type) + '_' + str(layer) + '_' + str(dataset) + '.npy'

    def _kr_initializer(_, dtype=None):
        weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', filename))
        return k.variable(value=weights, dtype=dtype)

    return _kr_initializer


def _kr_tf_weights_loader(dataset, weights_type, layer='DENSE'):
    filename = str(weights_type) + '_' + str(layer) + '_' + str(dataset) + '.npy'
    weights = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', filename))
    return weights


def get_classifier_tf(from_logits=False):
    """
    Standard TensorFlow classifier for unit testing.

    The following hyper-parameters were used to obtain the weights and biases:
    learning_rate: 0.01
    batch size: 10
    number of epochs: 2
    optimizer: tf.train.AdamOptimizer

    :return: TensorFlowClassifier, tf.Session()
    """
    # pylint: disable=E0401
    import tensorflow as tf
    if tf.__version__[0] == '2':
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
    from art.classifiers import TensorFlowClassifier

    # Define input and output placeholders
    input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    output_ph = tf.placeholder(tf.int32, shape=[None, 10])

    # Define the tensorflow graph
    conv = tf.layers.conv2d(input_ph, 1, 7, activation=tf.nn.relu,
                            kernel_initializer=_tf_weights_loader('MNIST', 'W', 'CONV2D'),
                            bias_initializer=_tf_weights_loader('MNIST', 'B', 'CONV2D'))
    conv = tf.layers.max_pooling2d(conv, 4, 4)
    flattened = tf.layers.flatten(conv)

    # Logits layer
    logits = tf.layers.dense(flattened, 10, kernel_initializer=_tf_weights_loader('MNIST', 'W', 'DENSE'),
                             bias_initializer=_tf_weights_loader('MNIST', 'B', 'DENSE'))

    # probabilities
    probabilities = tf.keras.activations.softmax(x=logits)

    # Train operator
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    # TensorFlow session and initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Create the classifier
    if from_logits:
        tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=logits, labels_ph=output_ph,
                                   train=train, loss=loss, learning=None, sess=sess)
    else:
        tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=probabilities, labels_ph=output_ph,
                                   train=train, loss=loss, learning=None, sess=sess)

    return tfc, sess


def get_classifier_tf_v2():
    """
    Standard TensorFlow v2 classifier for unit testing.

    The following hyper-parameters were used to obtain the weights and biases:
    learning_rate: 0.01
    batch size: 10
    number of epochs: 2
    optimizer: tf.train.AdamOptimizer

    :return: TensorFlowV2Classifier,
    """
    # pylint: disable=E0401
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
    from art.classifiers import TensorFlowV2Classifier

    if tf.__version__[0] != '2':
        raise ImportError('This function requires TensorFlow v2.')

    class TensorFlowModel(Model):
        """
        Standard TensorFlow model for unit testing
        """

        def __init__(self):
            super(TensorFlowModel, self).__init__()
            self.conv1 = Conv2D(filters=1, kernel_size=7, activation='relu',
                                kernel_initializer=_tf_weights_loader('MNIST', 'W', 'CONV2D', 2),
                                bias_initializer=_tf_weights_loader('MNIST', 'B', 'CONV2D', 2))
            self.maxpool = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid', data_format=None)
            self.flatten = Flatten()
            self.dense1 = Dense(10, activation='softmax',
                                kernel_initializer=_tf_weights_loader('MNIST', 'W', 'DENSE', 2),
                                bias_initializer=_tf_weights_loader('MNIST', 'B', 'DENSE', 2))

        def call(self, x):
            """
            Call function to evaluate the model

            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.dense1(x)
            return x

    model = TensorFlowModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Create the classifier
    tfc = TensorFlowV2Classifier(model=model, loss_object=loss_object, nb_classes=10, input_shape=(28, 28, 1),
                                 clip_values=(0, 1))

    return tfc


def get_classifier_kr(loss_name='categorical_crossentropy', loss_type='function_losses', from_logits=False):
    """
    Standard Keras classifier for unit testing

    The weights and biases are identical to the TensorFlow model in get_classifier_tf().

    :param loss_name: The name of the loss function.
    :type loss_name: `str`
    :param loss_type: The type of loss function definitions: label (loss function defined by string of its name),
                      function_losses (loss function imported from keras.losses), function_backend (loss function
                      imported from keras.backend)
    :type loss_type: `str`
    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`

    :return: KerasClassifier, tf.Session()
    """
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    from art.classifiers import KerasClassifier

    # Create simple CNN
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1),
                     kernel_initializer=_kr_weights_loader('MNIST', 'W', 'CONV2D'),
                     bias_initializer=_kr_weights_loader('MNIST', 'B', 'CONV2D')))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())

    if from_logits:
        model.add(Dense(10, activation='linear', kernel_initializer=_kr_weights_loader('MNIST', 'W', 'DENSE'),
                        bias_initializer=_kr_weights_loader('MNIST', 'B', 'DENSE')))
    else:
        model.add(Dense(10, activation='softmax', kernel_initializer=_kr_weights_loader('MNIST', 'W', 'DENSE'),
                        bias_initializer=_kr_weights_loader('MNIST', 'B', 'DENSE')))

    if loss_name == 'categorical_hinge':
        if loss_type == 'label':
            raise AttributeError('This combination of loss function options is not supported.')
        elif loss_type == 'function_losses':
            loss = keras.losses.categorical_hinge
    elif loss_name == 'categorical_crossentropy':
        if loss_type == 'label':
            if from_logits:
                raise AttributeError('This combination of loss function options is not supported.')
            else:
                loss = loss_name
        elif loss_type == 'function_losses':
            if from_logits:
                if int(keras.__version__.split('.')[0]) == 2 and int(keras.__version__.split('.')[1]) >= 3:
                    def categorical_crossentropy(y_true, y_pred):
                        return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

                    loss = categorical_crossentropy
                else:
                    raise AttributeError('This combination of loss function options is not supported.')
            else:
                loss = keras.losses.categorical_crossentropy
        elif loss_type == 'function_backend':
            if from_logits:
                def categorical_crossentropy(y_true, y_pred):
                    return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = categorical_crossentropy
            else:
                loss = keras.backend.categorical_crossentropy
    elif loss_name == 'sparse_categorical_crossentropy':
        if loss_type == 'label':
            if from_logits:
                raise AttributeError('This combination of loss function options is not supported.')
            else:
                loss = loss_name
        elif loss_type == 'function_losses':
            if from_logits:
                if int(keras.__version__.split('.')[0]) == 2 and int(keras.__version__.split('.')[1]) >= 3:
                    def sparse_categorical_crossentropy(y_true, y_pred):
                        return keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

                    loss = sparse_categorical_crossentropy
                else:
                    raise AttributeError('This combination of loss function options is not supported.')
            else:
                loss = keras.losses.sparse_categorical_crossentropy
        elif loss_type == 'function_backend':
            if from_logits:
                def sparse_categorical_crossentropy(y_true, y_pred):
                    return keras.backend.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = sparse_categorical_crossentropy
            else:
                loss = keras.backend.sparse_categorical_crossentropy
    elif loss_name == 'kullback_leibler_divergence':
        if loss_type == 'label':
            raise AttributeError('This combination of loss function options is not supported.')
        elif loss_type == 'function_losses':
            loss = keras.losses.kullback_leibler_divergence
        elif loss_type == 'function_backend':
            raise AttributeError('This combination of loss function options is not supported.')
    elif loss_name == 'cosine_similarity':
        if loss_type == 'label':
            loss = loss_name
        elif loss_type == 'function_losses':
            loss = keras.losses.cosine_similarity
        elif loss_type == 'function_backend':
            loss = keras.backend.cosine_similarity

    else:
        raise ValueError('Loss name not recognised.')

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=from_logits)

    return krc


def get_classifier_kr_tf(loss_name='categorical_crossentropy', loss_type='function', from_logits=False):
    """
    Standard Keras classifier for unit testing

    The weights and biases are identical to the TensorFlow model in get_classifier_tf().

    :param loss_name: The name of the loss function.
    :type loss_name: `str`
    :param loss_type: The type of loss function definitions: label (loss function defined by string of its name),
                      function_losses (loss function), class (loss function generator)
    :type loss_type: `str`
    :param from_logits: Flag if model should predict logits (True) or probabilities (False).
    :type from_logits: `bool`


    :return: KerasClassifier
    """
    # pylint: disable=E0401
    import tensorflow as tf
    if tf.__version__[0] == '2':
        tf.compat.v1.disable_eager_execution()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    from art.classifiers import KerasClassifier

    # Create simple CNN
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1)))
    model.layers[-1].set_weights(
        [_kr_tf_weights_loader('MNIST', 'W', 'CONV2D'), _kr_tf_weights_loader('MNIST', 'B', 'CONV2D')])
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())

    if from_logits:
        model.add(Dense(10, activation='linear'))
    else:
        model.add(Dense(10, activation='softmax'))

    model.layers[-1].set_weights(
        [_kr_tf_weights_loader('MNIST', 'W', 'DENSE'), _kr_tf_weights_loader('MNIST', 'B', 'DENSE')])

    if loss_name == 'categorical_hinge':
        if loss_type == 'label':
            loss = loss_name
        elif loss_type == 'function':
            loss = tf.keras.losses.categorical_hinge
        elif loss_type == 'class':
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError('This combination of loss function options is not supported.')
            loss = tf.keras.losses.CategoricalHinge(reduction=reduction)
    elif loss_name == 'categorical_crossentropy':
        if loss_type == 'label':
            if from_logits:
                raise AttributeError
            else:
                loss = loss_name
        elif loss_type == 'function':
            if from_logits:
                def categorical_crossentropy(y_true, y_pred):
                    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = categorical_crossentropy
            else:
                loss = tf.keras.losses.categorical_crossentropy
        elif loss_type == 'class':
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError('This combination of loss function options is not supported.')
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, reduction=reduction)
    elif loss_name == 'sparse_categorical_crossentropy':
        if loss_type == 'label':
            if from_logits:
                raise AttributeError
            else:
                loss = loss_name
        elif loss_type == 'function':
            if from_logits:
                def sparse_categorical_crossentropy(y_true, y_pred):
                    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

                loss = sparse_categorical_crossentropy
            else:
                loss = tf.keras.losses.sparse_categorical_crossentropy
        elif loss_type == 'class':
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError('This combination of loss function options is not supported.')
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction=reduction)
    elif loss_name == 'kullback_leibler_divergence':
        if loss_type == 'label':
            loss = loss_name
        elif loss_type == 'function':
            loss = tf.keras.losses.kullback_leibler_divergence
        elif loss_type == 'class':
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError('This combination of loss function options is not supported.')
            loss = tf.keras.losses.KLDivergence(reduction=reduction)
    elif loss_name == 'cosine_similarity':
        if loss_type == 'label':
            loss = loss_name
        elif loss_type == 'function':
            loss = tf.keras.losses.cosine_similarity
        elif loss_type == 'class':
            try:
                reduction = tf.keras.losses.Reduction.NONE
            except AttributeError:
                try:
                    reduction = tf.losses.Reduction.NONE
                except AttributeError:
                    try:
                        reduction = tf.python.keras.utils.losses_utils.ReductionV2.NONE
                    except AttributeError:
                        raise ImportError('This combination of loss function options is not supported.')
            loss = tf.keras.losses.CosineSimilarity(reduction=reduction)

    else:
        raise ValueError('Loss name not recognised.')

    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=from_logits)

    return krc


def get_classifier_kr_tf_binary():
    """
    Standard Tensorflow-Keras binary classifier for unit testing

    :return: KerasClassifier
    """
    # pylint: disable=E0401
    import tensorflow as tf
    if tf.__version__[0] == '2':
        tf.compat.v1.disable_eager_execution()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    from art.classifiers import KerasClassifier

    # Create simple CNN
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1)))
    model.layers[-1].set_weights(
        [_kr_tf_weights_loader('MNIST_BINARY', 'W', 'CONV2D'), _kr_tf_weights_loader('MNIST_BINARY', 'B', 'CONV2D')])
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.layers[-1].set_weights(
        [_kr_tf_weights_loader('MNIST_BINARY', 'W', 'DENSE'), _kr_tf_weights_loader('MNIST_BINARY', 'B', 'DENSE')])

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False)

    return krc


def get_classifier_pt(from_logits=False):
    """
    Standard PyTorch classifier for unit testing

    :return: PyTorchClassifier
    """
    from art.classifiers import PyTorchClassifier

    class Model(nn.Module):
        """
        Create model for pytorch.

        The weights and biases are identical to the TensorFlow model in get_classifier_tf().
        """

        def __init__(self):
            super(Model, self).__init__()

            w_conv2d = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'W_CONV2D_MNIST.npy'))
            b_conv2d = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'B_CONV2D_MNIST.npy'))
            w_dense = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'W_DENSE_MNIST.npy'))
            b_dense = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'B_DENSE_MNIST.npy'))

            w_conv2d_pt = w_conv2d.reshape((1, 1, 7, 7))

            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
            self.conv.weight = nn.Parameter(torch.Tensor(w_conv2d_pt))
            self.conv.bias = nn.Parameter(torch.Tensor(b_conv2d))
            self.pool = nn.MaxPool2d(4, 4)
            self.fullyconnected = nn.Linear(25, 10)
            self.fullyconnected.weight = nn.Parameter(torch.Tensor(np.transpose(w_dense)))
            self.fullyconnected.bias = nn.Parameter(torch.Tensor(b_dense))

        # pylint: disable=W0221
        # disable pylint because of API requirements for function
        def forward(self, x):
            """
            Forward function to evaluate the model
            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.conv(x)
            x = torch.nn.functional.relu(x)
            x = self.pool(x)
            x = x.reshape(-1, 25)
            x = self.fullyconnected(x)
            if not from_logits:
                x = torch.nn.functional.softmax(x)
            return x

    # Define the network
    model = Model()

    # Define a loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Get classifier
    ptc = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10,
                            clip_values=(0, 1))

    return ptc


def get_classifier_bb(defences=None):
    """
    Standard BlackBox classifier for unit testing

    :return: BlackBoxClassifier
    """
    from art.classifiers import BlackBoxClassifier
    from art.utils import to_categorical

    # define blackbox classifier
    def predict(x):
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'data/mnist', 'api_output.txt')) as json_file:
            predictions = json.load(json_file)
        return to_categorical(predictions['values'][:len(x)], nb_classes=10)

    bbc = BlackBoxClassifier(predict, (28, 28, 1), 10, clip_values=(0, 255), defences=defences)
    return bbc


def get_classifier_mx():
    """
    Standard MXNet classifier for unit testing

    :return: MXNetClassifier
    """
    import mxnet
    from mxnet.gluon.nn import Conv2D, MaxPool2D, Flatten, Dense
    from art.classifiers import MXClassifier

    model = mxnet.gluon.nn.Sequential()
    with model.name_scope():
        model.add(Conv2D(channels=1, kernel_size=7, activation='relu'),
                  MaxPool2D(pool_size=4, strides=4),
                  Flatten(),
                  Dense(10))
    model.initialize(init=mxnet.init.Xavier())

    # Create optimizer
    loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mxnet.gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})

    # # Fit classifier
    # classifier = MXClassifier(model=net, loss=loss, clip_values=(0, 1), input_shape=(1, 28, 28), nb_classes=10,
    #                           optimizer=trainer)
    # classifier.fit(x_train, y_train, batch_size=128, nb_epochs=2)
    # cls.classifier = classifier

    # Get classifier
    mxc = MXClassifier(model=model, loss=loss, input_shape=(28, 28, 1), nb_classes=10, optimizer=trainer, ctx=None,
                       channel_index=1, clip_values=(0, 1), defences=None, preprocessing=(0, 1))

    return mxc


# ------------------------------------------------------------------------------------------------ TEST MODELS FOR IRIS

def get_iris_classifier_tf():
    """
    Standard TensorFlow classifier for unit testing.

    The following hyper-parameters were used to obtain the weights and biases:

    * learning_rate: 0.01
    * batch size: 5
    * number of epochs: 200
    * optimizer: tf.train.AdamOptimizer

    The model is trained of 70% of the dataset, and 30% of the training set is used as validation split.

    :return: The trained model for Iris dataset and the session.
    :rtype: `tuple(TensorFlowClassifier, tf.Session)`
    """
    import tensorflow as tf
    if tf.__version__[0] == '2':
        # pylint: disable=E0401
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
    from art.classifiers import TensorFlowClassifier

    # Define input and output placeholders
    input_ph = tf.placeholder(tf.float32, shape=[None, 4])
    output_ph = tf.placeholder(tf.int32, shape=[None, 3])

    # Define the tensorflow graph
    dense1 = tf.layers.dense(input_ph, 10, kernel_initializer=_tf_weights_loader('IRIS', 'W', 'DENSE1'),
                             bias_initializer=_tf_weights_loader('IRIS', 'B', 'DENSE1'))
    dense2 = tf.layers.dense(dense1, 10, kernel_initializer=_tf_weights_loader('IRIS', 'W', 'DENSE2'),
                             bias_initializer=_tf_weights_loader('IRIS', 'B', 'DENSE2'))
    logits = tf.layers.dense(dense2, 3, kernel_initializer=_tf_weights_loader('IRIS', 'W', 'DENSE3'),
                             bias_initializer=_tf_weights_loader('IRIS', 'B', 'DENSE3'))

    # Train operator
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))

    # TensorFlow session and initialization
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train the classifier
    tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=logits, labels_ph=output_ph, train=None,
                               loss=loss, learning=None, sess=sess, channel_index=1)

    return tfc, sess


def get_iris_classifier_kr():
    """
    Standard Keras classifier for unit testing on Iris dataset. The weights and biases are identical to the TensorFlow
    model in `get_iris_classifier_tf`.

    :return: The trained model for Iris dataset and the session.
    :rtype: `tuple(KerasClassifier, tf.Session)`
    """
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    from art.classifiers import KerasClassifier

    # Create simple CNN
    model = Sequential()
    model.add(Dense(10, input_shape=(4,), activation='relu',
                    kernel_initializer=_kr_weights_loader('IRIS', 'W', 'DENSE1'),
                    bias_initializer=_kr_weights_loader('IRIS', 'B', 'DENSE1')))
    model.add(Dense(10, activation='relu', kernel_initializer=_kr_weights_loader('IRIS', 'W', 'DENSE2'),
                    bias_initializer=_kr_weights_loader('IRIS', 'B', 'DENSE2')))
    model.add(Dense(3, activation='softmax', kernel_initializer=_kr_weights_loader('IRIS', 'W', 'DENSE3'),
                    bias_initializer=_kr_weights_loader('IRIS', 'B', 'DENSE3')))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    # Get classifier
    krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False, channel_index=1)

    return krc


def get_iris_classifier_pt():
    """
    Standard PyTorch classifier for unit testing on Iris dataset.

    :return: Trained model for Iris dataset.
    :rtype: :class:`.PyTorchClassifier`
    """
    from art.classifiers import PyTorchClassifier

    class Model(nn.Module):
        """
        Create Iris model for PyTorch.

        The weights and biases are identical to the TensorFlow model in `get_iris_classifier_tf`.
        """

        def __init__(self):
            super(Model, self).__init__()

            w_dense1 = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'W_DENSE1_IRIS.npy'))
            b_dense1 = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'B_DENSE1_IRIS.npy'))
            w_dense2 = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'W_DENSE2_IRIS.npy'))
            b_dense2 = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'B_DENSE2_IRIS.npy'))
            w_dense3 = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'W_DENSE3_IRIS.npy'))
            b_dense3 = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'B_DENSE3_IRIS.npy'))

            self.fully_connected1 = nn.Linear(4, 10)
            self.fully_connected1.weight = nn.Parameter(torch.Tensor(np.transpose(w_dense1)))
            self.fully_connected1.bias = nn.Parameter(torch.Tensor(b_dense1))
            self.fully_connected2 = nn.Linear(10, 10)
            self.fully_connected2.weight = nn.Parameter(torch.Tensor(np.transpose(w_dense2)))
            self.fully_connected2.bias = nn.Parameter(torch.Tensor(b_dense2))
            self.fully_connected3 = nn.Linear(10, 3)
            self.fully_connected3.weight = nn.Parameter(torch.Tensor(np.transpose(w_dense3)))
            self.fully_connected3.bias = nn.Parameter(torch.Tensor(b_dense3))

        # pylint: disable=W0221
        # disable pylint because of API requirements for function
        def forward(self, x):
            x = self.fully_connected1(x)
            x = self.fully_connected2(x)
            logit_output = self.fully_connected3(x)

            return logit_output

    # Define the network
    model = Model()

    # Define a loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Get classifier
    ptc = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(4,), nb_classes=3,
                            clip_values=(0, 1), channel_index=1)

    return ptc
