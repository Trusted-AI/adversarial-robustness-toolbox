import unittest

import keras
from keras.utils import np_utils

import tensorflow as tf

import config
from cleverhans.utils_tf import model_train,model_eval

from src.classifiers import cnn
from src.utils import load_cifar10,load_mnist

class TestCNNModel(unittest.TestCase):

    def test_cifar(self):

        BATCH_SIZE = 10
        NB_CLASSES = 10
        NB_TRAIN = 1000
        NB_TEST = 100

        session = tf.Session()
        keras.backend.set_session(session)

        # get CIFAR10
        (X_train, Y_train), (X_test, Y_test) = load_cifar10()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(Y_train,NB_CLASSES)
        Y_test = np_utils.to_categorical(Y_test,NB_CLASSES)

        im_shape = X_train[0].shape

        x = tf.placeholder(tf.float32, shape=(None,im_shape[0],im_shape[1],im_shape[2]))
        y = tf.placeholder(tf.float32, shape=(None,NB_CLASSES))

        model = cnn.cnn_model(im_shape,act="brelu")
        predictions = model(x)

        def evaluate():
            # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
            eval_params = {'batch_size': BATCH_SIZE}
            accuracy = model_eval(session,x,y,predictions,X_test,Y_test,args=eval_params)

        train_params = {
            'nb_epochs': 1,
            'batch_size': BATCH_SIZE,
            'learning_rate': 0.1
        }
        model_train(session, x, y, predictions, X_train, Y_train, evaluate=evaluate, args=train_params)


    def test_mnist(self):

        BATCH_SIZE = 10
        NB_CLASSES = 10
        NB_TRAIN = 1000
        NB_TEST = 100

        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(Y_train,NB_CLASSES)
        Y_test = np_utils.to_categorical(Y_test,NB_CLASSES)

        im_shape = X_train[0].shape

        x = tf.placeholder(tf.float32, shape=(None,im_shape[0],im_shape[1],im_shape[2]))
        y = tf.placeholder(tf.float32, shape=(None,NB_CLASSES))

        model = cnn.cnn_model(im_shape,act="relu")
        predictions = model(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test examples
            eval_params = {'batch_size': BATCH_SIZE}
            accuracy = model_eval(session,x,y,predictions,X_test,Y_test,args=eval_params)

        train_params = {
            'nb_epochs': 1,
            'batch_size': BATCH_SIZE,
            'learning_rate': 0.1
        }
        model_train(session, x, y, predictions, X_train, Y_train, evaluate=evaluate, args=train_params)

if __name__ == '__main__':
    unittest.main()
