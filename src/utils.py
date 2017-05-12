from keras import backend as K

from keras.datasets.cifar import load_batch
from keras.utils import np_utils

import numpy as np
import os

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_cifar10():
    """Loads CIFAR10 dataset from config.CIFAR10_PATH.

    :return: `(x_train, y_train), (x_test, y_test)`
    :rtype: tuple of numpy.ndarray), (tuple of numpy.ndarray)
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

    x_train,y_train = preprocess(x_train,y_train)
    x_test,y_test = preprocess(x_test,y_test)

    return (x_train, y_train), (x_test, y_test)

def load_mnist():

    """Loads MNIST dataset from config.MNIST_PATH
    
    :return: `(x_train, y_train), (x_test, y_test)`
    :rtype: tuple of numpy.ndarray), (tuple of numpy.ndarray)
    """
    from config import MNIST_PATH

    f = np.load(os.path.join(MNIST_PATH, 'mnist.npz'))
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()

    # add channel axis
    x_train = np.expand_dims(x_train,axis=3)
    x_test = np.expand_dims(x_test,axis=3)

    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test)

def preprocess(x,y,nb_classes=10,max_value=255):
    """ Scales `x` to [0,1] and converts `y` to class matrices.
    
    :param x: array of instances
    :param y: array of labels
    :param int nb_classes: 
    :param int max_value: original maximal pixel value
    :return: x,y
    """

    x = x.astype('float32') / max_value
    y = np_utils.to_categorical(y,nb_classes)

    return x,y
