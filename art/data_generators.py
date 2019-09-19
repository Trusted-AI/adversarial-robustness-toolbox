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
Module defining an interface for data generators and providing concrete implementations for the supported frameworks.
Their purpose is to allow for data loading and batching on the fly, as well as dynamic data augmentation.
The generators can be used with the `fit_generator` function in the :class:`.Classifier` interface. Users can define
their own generators following the :class:`.DataGenerator` interface.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

import logging

logger = logging.getLogger(__name__)

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DataGenerator(ABC):
    """
    Base class for data generators.
    """

    def __init__(self, size, batch_size):
        """
        Base initializer for data generators.

        :param size: Total size of the dataset.
        :type size: `int` or `None`
        :param batch_size: Size of the minibatches.
        :type batch_size: `int`
        """
        if size is not None and (not isinstance(size, int) or size < 1):
            raise ValueError("The total size of the dataset must be an integer greater than zero.")
        self.size = size

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")
        self.batch_size = batch_size

        if size is not None and batch_size > size:
            raise ValueError("The batch size must be smaller than the dataset size.")

    @abc.abstractmethod
    def get_batch(self):
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        """
        raise NotImplementedError


class KerasDataGenerator(DataGenerator):
    """
    Wrapper class on top of the Keras-native data generators. These can either be generator functions,
    `keras.utils.Sequence` or Keras-specific data generators (`keras.preprocessing.image.ImageDataGenerator`).
    """

    def __init__(self, generator, size, batch_size):
        """
        Create a Keras data generator wrapper instance.

        :param generator: A generator as specified by Keras documentation. Its output must be a tuple of either
                          `(inputs, targets)` or `(inputs, targets, sample_weights)`. All arrays in this tuple must have
                          the same length. The generator is expected to loop over its data indefinitely.
        :type generator: generator function or `keras.utils.Sequence` or `keras.preprocessing.image.ImageDataGenerator`
        :param size: Total size of the dataset.
        :type size: `int` or `None`
        :param batch_size: Size of the minibatches.
        :type batch_size: `int`
        """
        super(KerasDataGenerator, self).__init__(size=size, batch_size=batch_size)
        self.generator = generator

    def get_batch(self):
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        """
        import inspect

        if inspect.isgeneratorfunction(self.generator):
            return next(self.generator)

        iter_ = iter(self.generator)
        return next(iter_)


class PyTorchDataGenerator(DataGenerator):
    """
    Wrapper class on top of the PyTorch native data loader :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, data_loader, size, batch_size):
        """
        Create a data generator wrapper on top of a PyTorch :class:`DataLoader`.

        :param data_loader: A PyTorch data generator.
        :type data_loader: `torch.utils.data.DataLoader`
        :param size: Total size of the dataset.
        :type size: int
        :param batch_size: Size of the minibatches.
        :type batch_size: int
        """
        super(PyTorchDataGenerator, self).__init__(size=size, batch_size=batch_size)

        from torch.utils.data import DataLoader

        if not isinstance(data_loader, DataLoader):
            raise TypeError('Expected instance of PyTorch `DataLoader, received %s instead.`' % str(type(data_loader)))

        self.data_loader = data_loader

    def get_batch(self):
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        """
        iter_ = iter(self.data_loader)
        batch = list(next(iter_))

        for i, item in enumerate(batch):
            batch[i] = item.data.cpu().numpy()

        return tuple(batch)


class MXDataGenerator(DataGenerator):
    """
    Wrapper class on top of the MXNet/Gluon native data loader :class:`mxnet.gluon.data.DataLoader`.
    """

    def __init__(self, data_loader, size, batch_size):
        """
        Create a data generator wrapper on top of an MXNet :class:`DataLoader`.

        :param data_loader:
        :type data_loader: `mxnet.gluon.data.DataLoader`
        :param size: Total size of the dataset.
        :type size: int
        :param batch_size: Size of the minibatches.
        :type batch_size: int
        """
        super(MXDataGenerator, self).__init__(size=size, batch_size=batch_size)

        from mxnet.gluon.data import DataLoader

        if not isinstance(data_loader, DataLoader):
            raise TypeError('Expected instance of Gluon `DataLoader, received %s instead.`' % str(type(data_loader)))

        self.data_loader = data_loader

    def get_batch(self):
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        """
        iter_ = iter(self.data_loader)
        batch = list(next(iter_))

        for i, item in enumerate(batch):
            batch[i] = item.asnumpy()

        return tuple(batch)


class TFDataGenerator(DataGenerator):
    """
    Wrapper class on top of the TensorFlow native iterators :class:`tf.data.Iterator`.
    """

    def __init__(self, sess, iterator, iterator_type, iterator_arg, size, batch_size):
        """
        Create a data generator wrapper for TensorFlow. Supported iterators: initializable, reinitializable, feedable.

        :param sess: TensorFlow session.
        :type sess: `tf.Session`
        :param iterator: Data iterator from TensorFlow.
        :type iterator: `tensorflow.python.data.ops.iterator_ops.Iterator`
        :param iterator_type: Type of the iterator. Supported types: `initializable`, `reinitializable`, `feedable`.
        :type iterator_type: `string`
        :param iterator_arg: Argument to initialize the iterator. It is either a feed_dict used for the initializable
        and feedable mode, or an init_op used for the reinitializable mode.
        :type iterator_arg: `dict`, `tuple` or `tensorflow.python.framework.ops.Operation`
        :param size: Total size of the dataset.
        :type size: `int`
        :param batch_size: Size of the minibatches.
        :type batch_size: `int`
        :raises: `TypeError`, `ValueError`
        """
        # pylint: disable=E0401
        import tensorflow as tf
        if tf.__version__[0] == '2':
            import tensorflow.compat.v1 as tf
            tf.disable_eager_execution()

        super(TFDataGenerator, self).__init__(size=size, batch_size=batch_size)
        self.sess = sess
        self.iterator = iterator
        self.iterator_type = iterator_type
        self.iterator_arg = iterator_arg

        if not isinstance(iterator, tf.data.Iterator):
            raise TypeError("Only support object tf.data.Iterator")

        if iterator_type == 'initializable':
            if not isinstance(iterator_arg, dict):
                raise TypeError("Need to pass a dictionary for iterator type %s" % iterator_type)
        elif iterator_type == 'reinitializable':
            if not isinstance(iterator_arg, tf.Operation):
                raise TypeError("Need to pass a tensorflow operation for iterator type %s" % iterator_type)
        elif iterator_type == 'feedable':
            if not isinstance(iterator_arg, tuple):
                raise TypeError("Need to pass a tuple for iterator type %s" % iterator_type)
        else:
            raise TypeError("Iterator type %s not supported" % iterator_type)

    def get_batch(self):
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        :raises: `ValueError` if the iterator has reached the end.
        """
        import tensorflow as tf

        # Get next batch
        next_batch = self.iterator.get_next()

        # Process to get the batch
        try:
            if self.iterator_type in ('initializable', 'reinitializable'):
                return self.sess.run(next_batch)
            return self.sess.run(next_batch, feed_dict=self.iterator_arg[1])
        except (tf.errors.FailedPreconditionError, tf.errors.OutOfRangeError):
            if self.iterator_type == 'initializable':
                self.sess.run(self.iterator.initializer, feed_dict=self.iterator_arg)
                return self.sess.run(next_batch)

            if self.iterator_type == 'reinitializable':
                self.sess.run(self.iterator_arg)
                return self.sess.run(next_batch)

            self.sess.run(self.iterator_arg[0].initializer)
            return self.sess.run(next_batch, feed_dict=self.iterator_arg[1])
