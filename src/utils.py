from __future__ import absolute_import

from keras.datasets.cifar import load_batch
from keras.datasets.mnist import load_data
from keras import backend as K

import numpy as np
import os


def load_cifar10():
    """Loads CIFAR10 dataset from config.CIFAR10_PATH.

    # Returns
        (tuple of numpy.ndarray), (tuple of numpy.ndarray): `(x_train, y_train), (x_test, y_test)`.
    """

    from config import CIFAR10_PATH

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(CIFAR10_PATH, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(CIFAR10_PATH, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def load_mnist():

    """Loads MNIST dataset from config.MNIST_PATH
    
    :# Returns
        (tuple of numpy.ndarray), (tuple of numpy.ndarray): `(x_train, y_train), (x_test, y_test)`.
    """
    from config import MNIST_PATH

    f = np.load(MNIST_PATH)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()

    # add channel axis
    x_train = np.expand_dims(x_train,axis=3)
    x_test = np.expand_dims(x_test,axis=3)

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    return (x_train, y_train), (x_test, y_test)