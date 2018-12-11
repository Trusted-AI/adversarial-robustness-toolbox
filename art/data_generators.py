"""
Module defining an interface for data generators and providing concrete implementations for the supported frameworks.
Their purpose is to allow for data loading and batching on the fly, as well as dynamic data augmentation.
The generators can be used with the `fit_generator` function in the :class:`Classifier` interface. Users can define
their own generators following the :class:`DataGenerator` interface.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Base class for data generators.
    """
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
    def __init__(self, generator):
        """
        Create a Keras data generator wrapper instance.

        :param generator: A generator as specified by Keras documentation. Its output must be a tuple of either
                          `(inputs, targets)` or `(inputs, targets, sample_weights)`. All arrays in this tuple must have
                          the same length. The generator is expected to loop over its data indefinitely.
        :type generator: generator function or `keras.utils.Sequence` or `keras.preprocessing.image.ImageDataGenerator`
        """
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
    def __init__(self, data_loader):
        """
        Create a data generator wrapper on top of a PyTorch :class:`DataLoader`.

        :param data_loader: A PyTorch data generator.
        :type data_loader: `torch.utils.data.DataLoader`
        """
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
    Wrapper class on top of the MXNet/Gluon native data loader :class:`mxnet.gluon.data.DataLoader``.
    """
    def __init__(self, data_loader):
        """
        Create a data generator wrapper on top of an MXNet :class:`DataLoader`.

        :param data_loader:
        :type data_loader: `mxnet.gluon.data.DataLoader`
        """
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
    # TODO Needs to cover QueueRunner and Coordinator, tf.train.shuffle_batch, tf.data.Dataset
    def __init__(self):
        pass

    def get_batch(self):
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        """
        raise NotImplementedError
