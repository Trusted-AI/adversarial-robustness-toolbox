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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from art.data_generators import KerasDataGenerator, PyTorchDataGenerator, MXDataGenerator, TensorFlowDataGenerator
from art.data_generators import TensorFlowV2DataGenerator, NumpyDataGenerator

from tests.utils import master_seed

logger = logging.getLogger(__name__)


class TestNumpyDataGenerator(unittest.TestCase):
    def setUp(self):
        self.m = 100
        self.n = (28, 28, 1)
        self.x = np.random.random((self.m,) + self.n)
        self.y = np.arange(self.m)
        self.batch_size = 30

    def test_gen_interface(self):
        data_generator = NumpyDataGenerator(
            self.x,
            self.y,
            batch_size=self.batch_size,
            drop_remainder=True,
            shuffle=False,
        )
        x_batch, y_batch = data_generator.get_batch()

        # Check return types
        self.assertTrue(isinstance(x_batch, np.ndarray))
        self.assertTrue(isinstance(y_batch, np.ndarray))

        # Check shapes
        self.assertEqual(x_batch.shape, (self.batch_size,) + self.n)
        self.assertEqual(y_batch.shape, (self.batch_size,))

        # Check underlying properties
        self.assertEqual(data_generator.size, len(self.x))
        self.assertEqual(data_generator.batch_size, self.batch_size)

    def test_errors(self):
        for x, y in [
            (1, [2, 3, 4]),  # x has no __len__
            ([1, 2], [3, 4, 5]),  # x and y are different lengths
        ]:
            with self.assertRaises(ValueError):
                NumpyDataGenerator(np.array(x), np.array(y))

    def test_drop_remainder_true(self):
        data_generator = NumpyDataGenerator(
            self.x,
            self.y,
            batch_size=self.batch_size,
            drop_remainder=True,
            shuffle=False,
        )
        self.assertEquals(data_generator.batches_per_epoch, 3)
        for i in range(3):
            data_generator.get_batch()
        _, y_batch = data_generator.get_batch()
        self.assertTrue((y_batch == self.y[: self.batch_size]).all())

    def test_nothing_to_drop(self):
        data_generator = NumpyDataGenerator(
            self.x,
            self.y,
            batch_size=10,
            drop_remainder=True,
            shuffle=False,
        )
        self.assertEquals(data_generator.batches_per_epoch, 10)
        for i in range(9):
            data_generator.get_batch()
        _, y_batch = data_generator.get_batch()
        self.assertEquals(len(y_batch), 10)

    def test_drop_remainder_false(self):
        data_generator = NumpyDataGenerator(
            self.x,
            self.y,
            batch_size=self.batch_size,
            drop_remainder=False,
            shuffle=False,
        )
        self.assertEquals(data_generator.batches_per_epoch, 4)
        for i in range(3):
            data_generator.get_batch()
        x_batch, _ = data_generator.get_batch()
        self.assertEquals(len(x_batch), self.m % self.batch_size)

    def test_shuffle(self):
        """
        NOTE: assertions may fail randomly, but with negligibly low probability
            on the order of 1 / (m choose batch_size)
            Setting the master seed here should avoid that, however
        """
        master_seed(seed=42)
        data_generator = NumpyDataGenerator(
            self.x,
            self.y,
            batch_size=self.batch_size,
            drop_remainder=True,
            shuffle=True,
        )
        _, y_batch = data_generator.get_batch()
        self.assertFalse((y_batch == self.y[: self.batch_size]).all())

        # Test shuffling across epochs
        for i in range(2):
            data_generator.get_batch()
        _, y_batch_epoch2 = data_generator.get_batch()
        self.assertFalse((y_batch == y_batch_epoch2).all())

    def test_single_epoch_generator(self):
        data_generator = NumpyDataGenerator(
            self.x,
            self.y,
            batch_size=self.batch_size,
            drop_remainder=True,
            shuffle=False,
        )
        self.assertEquals(data_generator.batches_per_epoch, 3)
        for i in range(3):
            data_generator.get_batch()
        _, y_batch = data_generator.get_batch()
        self.assertTrue((y_batch == self.y[: self.batch_size]).all())


class TestKerasDataGenerator(unittest.TestCase):
    def setUp(self):
        from tensorflow import keras

        master_seed(seed=42)

        class DummySequence(keras.utils.Sequence):
            def __init__(self):
                self._size = 5
                self._x = np.random.rand(self._size, 28, 28, 1)
                self._y = np.random.randint(0, high=10, size=(self._size, 10))

            def __len__(self):
                return self._size

            def __getitem__(self, idx):
                return self._x[idx], self._y[idx]

        sequence = DummySequence()
        self.data_gen = KerasDataGenerator(sequence, size=5, batch_size=1)

    def tearDown(self):
        import keras.backend as k

        k.clear_session()

    def test_gen_interface(self):
        gen = self._dummy_gen()
        data_gen = KerasDataGenerator(gen, size=None, batch_size=5)

        x, y = data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 28, 28, 1))
        self.assertEqual(y.shape, (5, 10))

    def test_gen_keras_specific(self):
        gen = self._dummy_gen()
        data_gen = KerasDataGenerator(gen, size=None, batch_size=5)

        iter_ = iter(data_gen.iterator)
        x, y = next(iter_)

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 28, 28, 1))
        self.assertEqual(y.shape, (5, 10))

    def test_sequence_keras_specific(self):
        iter_ = iter(self.data_gen.iterator)
        x, y = next(iter_)

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (28, 28, 1))
        self.assertEqual(y.shape, (10,))

    def test_sequence_interface(self):
        x, y = self.data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (28, 28, 1))
        self.assertEqual(y.shape, (10,))

    def test_imagedatagen_interface(self):
        train_size, batch_size = 20, 5
        x_train, y_train = np.random.rand(train_size, 28, 28, 1), np.random.randint(0, 2, size=(train_size, 10))

        datagen = ImageDataGenerator(
            width_shift_range=0.075,
            height_shift_range=0.075,
            rotation_range=12,
            shear_range=0.075,
            zoom_range=0.05,
            fill_mode="constant",
            cval=0,
        )
        datagen.fit(x_train)

        # Create generator and get batch
        data_gen = KerasDataGenerator(
            datagen.flow(x_train, y_train, batch_size=batch_size), size=None, batch_size=batch_size
        )
        x, y = data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (batch_size, 28, 28, 1))
        self.assertEqual(y.shape, (batch_size, 10))

    def test_imagedatagen_keras_specific(self):
        train_size, batch_size = 20, 5
        x_train, y_train = np.random.rand(train_size, 28, 28, 1), np.random.randint(0, 2, size=(train_size, 10))

        datagen = ImageDataGenerator(
            width_shift_range=0.075,
            height_shift_range=0.075,
            rotation_range=12,
            shear_range=0.075,
            zoom_range=0.05,
            fill_mode="constant",
            cval=0,
        )
        datagen.fit(x_train)

        # Create generator and get batch
        data_gen = KerasDataGenerator(
            datagen.flow(x_train, y_train, batch_size=batch_size), size=None, batch_size=batch_size
        )
        x, y = next(data_gen.iterator)

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (batch_size, 28, 28, 1))
        self.assertEqual(y.shape, (batch_size, 10))

    @staticmethod
    def _dummy_gen(size=5):
        yield np.random.rand(size, 28, 28, 1), np.random.randint(low=0, high=10, size=(size, 10))


class TestPyTorchGenerator(unittest.TestCase):
    def setUp(self):
        import torch
        from torch.utils.data import DataLoader

        master_seed(seed=42)

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self._size = 10
                self._x = np.random.rand(self._size, 1, 5, 5)
                self._y = np.random.randint(0, high=10, size=self._size)

            def __len__(self):
                return self._size

            def __getitem__(self, idx):
                return self._x[idx], self._y[idx]

        dataset = DummyDataset()
        data_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
        self.data_gen = PyTorchDataGenerator(data_loader, size=10, batch_size=5)

    def test_gen_interface(self):
        x, y = self.data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 1, 5, 5))
        self.assertEqual(y.shape, (5,))

    def test_pytorch_specific(self):
        import torch

        iter_ = iter(self.data_gen.iterator)
        x, y = next(iter_)

        # Check return types
        self.assertTrue(isinstance(x, torch.Tensor))
        self.assertTrue(isinstance(y, torch.Tensor))

        # Check shapes
        self.assertEqual(x.shape, (5, 1, 5, 5))
        self.assertEqual(y.shape, (5,))

    def test_error(self):
        with self.assertRaises(TypeError):
            self.data_gen = PyTorchDataGenerator("data_loader", size=10, batch_size=5)


class TestMXGenerator(unittest.TestCase):
    def setUp(self):
        import mxnet as mx

        master_seed(seed=42, set_mxnet=True)

        x = mx.random.uniform(shape=(10, 1, 5, 5))
        y = mx.random.uniform(shape=10)
        dataset = mx.gluon.data.dataset.ArrayDataset(x, y)

        data_loader = mx.gluon.data.DataLoader(dataset, batch_size=5, shuffle=True)
        self.data_gen = MXDataGenerator(data_loader, size=10, batch_size=5)

    def test_gen_interface(self):
        x, y = self.data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 1, 5, 5))
        self.assertEqual(y.shape, (5,))

    def test_mxnet_specific(self):
        import mxnet as mx

        iter_ = iter(self.data_gen.iterator)
        x, y = next(iter_)

        # Check return types
        self.assertTrue(isinstance(x, mx.ndarray.NDArray))
        self.assertTrue(isinstance(y, mx.ndarray.NDArray))

        # Check shapes
        self.assertEqual(x.shape, (5, 1, 5, 5))
        self.assertEqual(y.shape, (5,))

    def test_error(self):
        with self.assertRaises(TypeError):
            _ = MXDataGenerator("data_loader", size=10, batch_size=5)


@unittest.skipIf(tf.__version__[0] == "2", reason="Skip unittests for TensorFlow v2.")
class TestTensorFlowDataGenerator(unittest.TestCase):
    def setUp(self):
        master_seed(seed=42)

        def generator(batch_size=5):
            while True:
                yield np.random.rand(batch_size, 5, 5, 1), np.random.randint(0, 10, size=10 * batch_size).reshape(
                    batch_size, -1
                )

        self.sess = tf.Session()
        self.dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.int32))

    def tearDown(self):
        self.sess.close()

    def test_init(self):
        iter_ = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        data_gen = TensorFlowDataGenerator(
            sess=self.sess, iterator=iter_, iterator_type="initializable", iterator_arg={}, size=10, batch_size=5
        )
        x, y = data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 5, 5, 1))
        self.assertEqual(y.shape, (5, 10))

    def test_reinit(self):
        iter_ = tf.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(self.dataset), tf.compat.v1.data.get_output_shapes(self.dataset)
        )
        init_op = iter_.make_initializer(self.dataset)
        data_gen = TensorFlowDataGenerator(
            sess=self.sess, iterator=iter_, iterator_type="reinitializable", iterator_arg=init_op, size=10, batch_size=5
        )
        x, y = data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 5, 5, 1))
        self.assertEqual(y.shape, (5, 10))

    def test_feedable(self):
        handle = tf.placeholder(tf.string, shape=[])
        iter_ = tf.data.Iterator.from_string_handle(
            handle, tf.compat.v1.data.get_output_types(self.dataset), tf.compat.v1.data.get_output_shapes(self.dataset)
        )
        feed_iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        feed_handle = self.sess.run(feed_iterator.string_handle())
        data_gen = TensorFlowDataGenerator(
            sess=self.sess,
            iterator=iter_,
            iterator_type="feedable",
            iterator_arg=(feed_iterator, {handle: feed_handle}),
            size=10,
            batch_size=5,
        )
        x, y = data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 5, 5, 1))
        self.assertEqual(y.shape, (5, 10))

    def test_error(self):
        handle = tf.placeholder(tf.string, shape=[])
        iter_ = tf.data.Iterator.from_string_handle(
            handle, tf.compat.v1.data.get_output_types(self.dataset), tf.compat.v1.data.get_output_shapes(self.dataset)
        )
        feed_iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        feed_handle = self.sess.run(feed_iterator.string_handle())

        with self.assertRaises(TypeError):
            _ = TensorFlowDataGenerator(
                sess=self.sess,
                iterator="iter_",
                iterator_type="feedable",
                iterator_arg=(feed_iterator, {handle: feed_handle}),
                size=10,
                batch_size=5,
            )

        with self.assertRaises(TypeError):
            _ = TensorFlowDataGenerator(
                sess=self.sess,
                iterator=iter_,
                iterator_type="initializable",
                iterator_arg=(feed_iterator, "{handle: feed_handle}"),
                size=10,
                batch_size=5,
            )

        with self.assertRaises(TypeError):
            _ = TensorFlowDataGenerator(
                sess=self.sess,
                iterator=iter_,
                iterator_type="reinitializable",
                iterator_arg=(feed_iterator, "{handle: feed_handle}"),
                size=10,
                batch_size=5,
            )

        with self.assertRaises(TypeError):
            _ = TensorFlowDataGenerator(
                sess=self.sess,
                iterator=iter_,
                iterator_type="feedable",
                iterator_arg=[feed_iterator, {handle: feed_handle}],
                size=10,
                batch_size=5,
            )

        with self.assertRaises(TypeError):
            _ = TensorFlowDataGenerator(
                sess=self.sess,
                iterator=iter_,
                iterator_type="test",
                iterator_arg=(feed_iterator, {handle: feed_handle}),
                size=10,
                batch_size=5,
            )


@unittest.skipIf(tf.__version__[0] == "1", reason="Skip unittests for TensorFlow v1.")
class TestTensorFlowV2DataGenerator(unittest.TestCase):
    def setUp(self):
        master_seed(seed=42)
        self.batch_size = 5
        x = np.random.rand(self.batch_size, 5, 5, 1)
        y = np.random.randint(0, 10, size=10 * self.batch_size).reshape(self.batch_size, -1)

        self.dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(100).batch(self.batch_size)

    def test_init(self):
        data_gen = TensorFlowV2DataGenerator(iterator=self.dataset, size=5, batch_size=self.batch_size)
        x, y = data_gen.get_batch()

        # Check return types
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

        # Check shapes
        self.assertEqual(x.shape, (5, 5, 5, 1))
        self.assertEqual(y.shape, (5, 10))

    def test_error(self):
        with self.assertRaises(TypeError):
            _ = TensorFlowV2DataGenerator(iterator="self.dataset", size=5, batch_size=self.batch_size)

        with self.assertRaises(ValueError):
            _ = TensorFlowV2DataGenerator(iterator=self.dataset, size=-5, batch_size=self.batch_size)

        with self.assertRaises(ValueError):
            _ = TensorFlowV2DataGenerator(iterator=self.dataset, size=5, batch_size=-self.batch_size)

        with self.assertRaises(ValueError):
            _ = TensorFlowV2DataGenerator(iterator=self.dataset, size=1, batch_size=self.batch_size)


if __name__ == "__main__":
    unittest.main()
