# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
their own generators following the :class:`.DataGenerator` interface. For large, numpy array-based  datasets, the
:class:`.NumpyDataGenerator` class can be flexibly used with `fit_generator` on framework-specific classifiers.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import inspect
import logging
from typing import Any, Dict, Generator, Iterator, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import keras
    import mxnet
    import tensorflow as tf
    import torch

logger = logging.getLogger(__name__)


class DataGenerator(abc.ABC):
    """
    Base class for data generators.
    """

    def __init__(self, size: Optional[int], batch_size: int) -> None:
        """
        Base initializer for data generators.

        :param size: Total size of the dataset.
        :param batch_size: Size of the minibatches.
        """
        if size is not None and (not isinstance(size, int) or size < 1):
            raise ValueError("The total size of the dataset must be an integer greater than zero.")
        self._size = size

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")
        self._batch_size = batch_size

        if size is not None and batch_size > size:
            raise ValueError("The batch size must be smaller than the dataset size.")

        self._iterator: Optional[Any] = None

    @abc.abstractmethod
    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        """
        raise NotImplementedError

    @property
    def iterator(self):
        """
        :return: Return the framework's iterable data generator.
        """
        return self._iterator

    @property
    def batch_size(self) -> int:
        """
        :return: Return the batch size.
        """
        return self._batch_size

    @property
    def size(self) -> Optional[int]:
        """
        :return: Return the dataset size.
        """
        return self._size


class NumpyDataGenerator(DataGenerator):
    """
    Simple numpy data generator backed by numpy arrays.

    Can be useful for applying numpy data to estimators in other frameworks
        e.g., when translating the entire numpy data to GPU tensors would cause OOM
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 1,
        drop_remainder: bool = True,
        shuffle: bool = False,
    ):
        """
        Create a numpy data generator backed by numpy arrays

        :param x: Numpy array of inputs
        :param y: Numpy array of targets
        :param batch_size: Size of the minibatches
        :param drop_remainder: Whether to omit the last incomplete minibatch in an epoch
        :param shuffle: Whether to shuffle the dataset for each epoch
        """
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        try:
            if len(x) != len(y):
                raise ValueError("inputs must be of equal length")
        except TypeError as err:
            raise ValueError(f"inputs x {x} and y {y} must be sized objects") from err
        size = len(x)
        self.x = x
        self.y = y
        super().__init__(size, int(batch_size))
        self.shuffle = bool(shuffle)

        self.drop_remainder = bool(drop_remainder)
        batches_per_epoch = size / self.batch_size
        if not self.drop_remainder:
            batches_per_epoch = np.ceil(batches_per_epoch)
        self.batches_per_epoch = int(batches_per_epoch)
        self._iterator = self
        self.generator: Iterator[Any] = iter([])

    def __iter__(self):
        if self.shuffle:
            index = np.arange(self.size)
            np.random.shuffle(index)
            for i in range(self.batches_per_epoch):
                batch_index = index[i * self.batch_size : (i + 1) * self.batch_size]
                yield (self.x[batch_index], self.y[batch_index])
        else:
            for i in range(self.batches_per_epoch):
                yield (
                    self.x[i * self.batch_size : (i + 1) * self.batch_size],
                    self.y[i * self.batch_size : (i + 1) * self.batch_size],
                )

    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`.
            The generator will loop over the data indefinitely.
            If drop_remainder is True, then the last minibatch in each epoch may be a different size

        :return: A tuple containing a batch of data `(x, y)`.
        """
        try:
            return next(self.generator)
        except StopIteration:
            self.generator = iter(self)
            return next(self.generator)


class KerasDataGenerator(DataGenerator):
    """
    Wrapper class on top of the Keras-native data generators. These can either be generator functions,
    `keras.utils.Sequence` or Keras-specific data generators (`keras.preprocessing.image.ImageDataGenerator`).
    """

    def __init__(
        self,
        iterator: Union[
            "keras.utils.Sequence",
            "tf.keras.utils.Sequence",
            "keras.preprocessing.image.ImageDataGenerator",
            "tf.keras.preprocessing.image.ImageDataGenerator",
            Generator,
        ],
        size: Optional[int],
        batch_size: int,
    ) -> None:
        """
        Create a Keras data generator wrapper instance.

        :param iterator: A generator as specified by Keras documentation. Its output must be a tuple of either
                         `(inputs, targets)` or `(inputs, targets, sample_weights)`. All arrays in this tuple must have
                         the same length. The generator is expected to loop over its data indefinitely.
        :param size: Total size of the dataset.
        :param batch_size: Size of the minibatches.
        """
        super().__init__(size=size, batch_size=batch_size)
        self._iterator = iterator

    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        """
        if inspect.isgeneratorfunction(self.iterator):
            return next(self.iterator)

        iter_ = iter(self.iterator)
        return next(iter_)


class PyTorchDataGenerator(DataGenerator):
    """
    Wrapper class on top of the PyTorch native data loader :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, iterator: "torch.utils.data.DataLoader", size: int, batch_size: int) -> None:
        """
        Create a data generator wrapper on top of a PyTorch :class:`DataLoader`.

        :param iterator: A PyTorch data generator.
        :param size: Total size of the dataset.
        :param batch_size: Size of the minibatches.
        """
        from torch.utils.data import DataLoader

        super().__init__(size=size, batch_size=batch_size)
        if not isinstance(iterator, DataLoader):
            raise TypeError(f"Expected instance of PyTorch `DataLoader, received {type(iterator)} instead.`")

        self._iterator: DataLoader = iterator
        self._current = iter(self.iterator)

    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        """
        try:
            batch = list(next(self._current))
        except StopIteration:
            self._current = iter(self.iterator)
            batch = list(next(self._current))

        for i, item in enumerate(batch):
            batch[i] = item.data.cpu().numpy()

        return tuple(batch)


class MXDataGenerator(DataGenerator):
    """
    Wrapper class on top of the MXNet/Gluon native data loader :class:`mxnet.gluon.data.DataLoader`.
    """

    def __init__(self, iterator: "mxnet.gluon.data.DataLoader", size: int, batch_size: int) -> None:
        """
        Create a data generator wrapper on top of an MXNet :class:`DataLoader`.

        :param iterator: A MXNet DataLoader instance.
        :param size: Total size of the dataset.
        :param batch_size: Size of the minibatches.
        """
        import mxnet

        super().__init__(size=size, batch_size=batch_size)
        if not isinstance(iterator, mxnet.gluon.data.DataLoader):
            raise TypeError(f"Expected instance of Gluon `DataLoader, received {type(iterator)} instead.`")

        self._iterator = iterator
        self._current = iter(self.iterator)

    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        """
        try:
            batch = list(next(self._current))
        except StopIteration:
            self._current = iter(self.iterator)
            batch = list(next(self._current))

        for i, item in enumerate(batch):
            batch[i] = item.asnumpy()

        return tuple(batch)


class TensorFlowDataGenerator(DataGenerator):  # pragma: no cover
    """
    Wrapper class on top of the TensorFlow native iterators :class:`tf.data.Iterator`.
    """

    def __init__(
        self,
        sess: "tf.Session",
        iterator: "tf.data.Iterator",
        iterator_type: str,
        iterator_arg: Union[Dict, Tuple, "tf.Operation"],
        size: int,
        batch_size: int,
    ) -> None:
        """
        Create a data generator wrapper for TensorFlow. Supported iterators: initializable, reinitializable, feedable.

        :param sess: TensorFlow session.
        :param iterator: Data iterator from TensorFlow.
        :param iterator_type: Type of the iterator. Supported types: `initializable`, `reinitializable`, `feedable`.
        :param iterator_arg: Argument to initialize the iterator. It is either a feed_dict used for the initializable
        and feedable mode, or an init_op used for the reinitializable mode.
        :param size: Total size of the dataset.
        :param batch_size: Size of the minibatches.
        :raises `TypeError`, `ValueError`: If input parameters are not valid.
        """
        # pylint: disable=E0401
        import tensorflow.compat.v1 as tf

        super().__init__(size=size, batch_size=batch_size)
        self.sess = sess
        self._iterator = iterator
        self.iterator_type = iterator_type
        self.iterator_arg = iterator_arg

        if not isinstance(iterator, tf.data.Iterator):
            raise TypeError("Only support object tf.data.Iterator")

        if iterator_type == "initializable":
            if not isinstance(iterator_arg, dict):
                raise TypeError(f"Need to pass a dictionary for iterator type {iterator_type}")
        elif iterator_type == "reinitializable":
            if not isinstance(iterator_arg, tf.Operation):
                raise TypeError(f"Need to pass a TensorFlow operation for iterator type {iterator_type}")
        elif iterator_type == "feedable":
            if not isinstance(iterator_arg, tuple):
                raise TypeError(f"Need to pass a tuple for iterator type {iterator_type}")
        else:
            raise TypeError(f"Iterator type {iterator_type} not supported")

    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :raises `ValueError`: If the iterator has reached the end.
        """
        import tensorflow as tf

        # Get next batch
        next_batch = self.iterator.get_next()

        # Process to get the batch
        try:
            if self.iterator_type in ("initializable", "reinitializable"):
                return self.sess.run(next_batch)
            return self.sess.run(next_batch, feed_dict=self.iterator_arg[1])
        except (tf.errors.FailedPreconditionError, tf.errors.OutOfRangeError):
            if self.iterator_type == "initializable":
                self.sess.run(self.iterator.initializer, feed_dict=self.iterator_arg)
                return self.sess.run(next_batch)

            if self.iterator_type == "reinitializable":
                self.sess.run(self.iterator_arg)
                return self.sess.run(next_batch)

            self.sess.run(self.iterator_arg[0].initializer)
            return self.sess.run(next_batch, feed_dict=self.iterator_arg[1])


class TensorFlowV2DataGenerator(DataGenerator):
    """
    Wrapper class on top of the TensorFlow v2 native iterators :class:`tf.data.Iterator`.
    """

    def __init__(self, iterator: "tf.data.Dataset", size: int, batch_size: int) -> None:
        """
        Create a data generator wrapper for TensorFlow. Supported iterators: initializable, reinitializable, feedable.

        :param iterator: TensorFlow Dataset.
        :param size: Total size of the dataset.
        :param batch_size: Size of the minibatches.
        :raises `TypeError`, `ValueError`: If input parameters are not valid.
        """
        # pylint: disable=E0401
        import tensorflow as tf

        super().__init__(size=size, batch_size=batch_size)
        self._iterator = iterator
        self._iterator_iter = iter(iterator)

        if not isinstance(iterator, tf.data.Dataset):
            raise TypeError("Only support object tf.data.Dataset")

    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :raises `ValueError`: If the iterator has reached the end.
        """
        # Get next batch
        x, y = next(self._iterator_iter)
        return x.numpy(), y.numpy()
