from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import keras.backend as k
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten
import numpy as np
import unittest

from art.attacks.carlini import CarliniL2Method
from art.classifiers import EnsembleClassifier, KerasClassifier
from art.utils import load_mnist, random_targets

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestEnsembleClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.clear_session()
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = ((x_train, y_train), (x_test, y_test))
        im_shape = x_train[0].shape

        # Create basic CNN on MNIST; architecture from Keras examples
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        cls.model_mnist = model

        import requests
        import tempfile
        import os

        # Temporary folder for tests
        cls.test_dir = tempfile.mkdtemp()

        # Download one ImageNet pic for tests
        url = 'http://farm1.static.flickr.com/163/381342603_81db58bea4.jpg'
        result = requests.get(url, stream=True)
        if result.status_code == 200:
            image = result.raw.read()
            open(os.path.join(cls.test_dir, 'test.jpg'), 'wb').write(image)

    @classmethod
    def tearDownClass(cls):
        k.clear_session()

        import shutil
        shutil.rmtree(cls.test_dir)

    # def test_logits(self):
    #     classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=True)

    # def test_probabilities(self):
    #     classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=False)

    @staticmethod
    def functional_model():
        in_layer = Input(shape=(28, 28, 1), name="input0")
        layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(in_layer)
        layer = Conv2D(64, (3, 3), activation='relu')(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)
        layer = Dropout(0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation='relu')(layer)
        layer = Dropout(0.5)(layer)
        out_layer = Dense(10, activation='softmax', name="output0")(layer)

        in_layer_2 = Input(shape=(28, 28, 1), name="input1")
        layer = Conv2D(32, kernel_size=(3, 3), activation='relu')(in_layer_2)
        layer = Conv2D(64, (3, 3), activation='relu')(layer)
        layer = MaxPooling2D(pool_size=(2, 2))(layer)
        layer = Dropout(0.25)(layer)
        layer = Flatten()(layer)
        layer = Dense(128, activation='relu')(layer)
        layer = Dropout(0.5)(layer)
        out_layer_2 = Dense(10, activation='softmax', name="output1")(layer)

        model = Model(inputs=[in_layer, in_layer_2], outputs=[out_layer, out_layer_2])

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'], loss_weights=[1., 1.0])

        return model

    def test_fit(self):
        self._test_fit(custom_activation=False)
        self._test_fit(custom_activation=True)

    def _test_fit(self, custom_activation=False):
        labels = np.argmax(self.mnist[1][1], axis=1)
        classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=False, custom_activation=custom_activation)
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        print("\nAccuracy: %.2f%%" % (acc * 100))

        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        print("\nAccuracy: %.2f%%" % (acc2 * 100))

        self.assertTrue(acc2 >= .9 * acc)

    def test_shapes(self):
        self._test_shapes(custom_activation=True)
        self._test_shapes(custom_activation=False)

    def _test_shapes(self, custom_activation=False):
        x_test, y_test = self.mnist[1]
        classifier = KerasClassifier((0, 1), self.model_mnist, custom_activation=custom_activation)

        preds = classifier.predict(self.mnist[1][0])
        self.assertTrue(preds.shape == y_test.shape)

        self.assertTrue(classifier.nb_classes == 10)

        class_grads = classifier.class_gradient(x_test[:11])
        self.assertTrue(class_grads.shape == tuple([11, 10] + list(x_test[1].shape)))

        loss_grads = classifier.loss_gradient(x_test[:11], y_test[:11])
        self.assertTrue(loss_grads.shape == x_test[:11].shape)

    def test_functional_model(self):
        self._test_functional_model(custom_activation=True)
        self._test_functional_model(custom_activation=False)

    def _test_functional_model(self, custom_activation=True):
        # Need to update the functional_model code to produce a model with more than one input and output layers...
        m = self.functional_model()
        keras_model = KerasClassifier((0, 1), m, input_layer=1, output_layer=1, custom_activation=custom_activation)
        self.assertTrue(keras_model._input.name, "input1")
        self.assertTrue(keras_model._output.name, "output1")
        keras_model = KerasClassifier((0, 1), m, input_layer=0, output_layer=0, custom_activation=custom_activation)
        self.assertTrue(keras_model._input.name, "input0")
        self.assertTrue(keras_model._output.name, "output0")

    def test_layers(self):
        self._test_layers(custom_activation=False)
        self._test_layers(custom_activation=True)

    def _test_layers(self, custom_activation=False):
        # Get MNIST
        (_, _), (x_test, _), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]

        classifier = KerasClassifier((0, 1), model=self.model_mnist, custom_activation=custom_activation)
        self.assertEqual(len(classifier.layer_names), 5)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            if 'dropout' not in name:
                act_i = classifier.get_activations(x_test, i)
                act_name = classifier.get_activations(x_test, name)
                self.assertAlmostEqual(np.sum(act_name - act_i), 0)

        self.assertTrue(classifier.get_activations(x_test, 0).shape == (NB_TEST, 26, 26, 32))
        self.assertTrue(classifier.get_activations(x_test, 4).shape == (NB_TEST, 128))

    def test_resnet(self):
        self._test_resnet(custom_activation=False)
        self._test_resnet(custom_activation=True)

    def _test_resnet(self, custom_activation=False):
        import os

        from keras.applications.resnet50 import ResNet50, decode_predictions
        from keras.preprocessing.image import load_img, img_to_array

        keras.backend.set_learning_phase(0)
        model = ResNet50(weights='imagenet')
        classifier = KerasClassifier((0, 255), model, custom_activation=custom_activation)

        # Load image from file
        image = img_to_array(load_img(os.path.join(self.test_dir, 'test.jpg'), target_size=(224, 224)))
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        label = decode_predictions(classifier.predict(image))[0][0]
        self.assertEqual(label[1], 'Weimaraner')
    
    @staticmethod
    def __get_model(cls, epochs=1):
        im_shape = cls.mnist[0][0][0].shape
        # Create basic CNN on MNIST; architecture from Keras examples
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(cls.mnist[0][0], cls.mnist[0][1], batch_size=BATCH_SIZE, epochs=epochs)
        return model

    def test_create_ensemble(self):
        model_1 = KerasClassifier((0,1), self.__get_model(self,epochs=2))
        model_2 = KerasClassifier((0,1), self.__get_model(self,epochs=2))
        ensemble = EnsembleClassifier((0,1), model_1, model_2)
        y = np.zeros((1,10))
        y[:,0] = 1
        grad = ensemble.loss_gradient(self.mnist[0][0][0:1], y)
        print(grad.shape)
        grad = ensemble.loss_gradient(self.mnist[0][0][0:1], y, raw=True)
        print(grad.shape)

        cl2m = CarliniL2Method(classifier=ensemble, targeted=True, max_iter=200, binary_search_steps=10,
                               learning_rate=1, initial_const=1, decay=0., confidence=0)
        x_test = self.mnist[1][0][:2]
        y_test = self.mnist[1][1][:2]
        params = {'y': random_targets(y_test, ensemble.nb_classes), 'confidence' : 0.0}
        x_test_adv = cl2m.generate(x_test, **params)
        o1 = np.argmax(model_1.predict(x_test_adv), axis=1)
        o2 = np.argmax(model_2.predict(x_test_adv), axis=1)
        e = np.argmax(ensemble.predict(x_test_adv), axis=1)
        print('Param:', params)
        print('Truth:',np.argmax(y_test, axis=1))
        print('Target:',np.argmax(params['y'], axis=1))
        print('Observed:',o1,o2,e)
        print("Changed:",(x_test == x_test_adv).any())
    
    @staticmethod
    def __get_multi_output_model():
        # Create one model with two or more outputs, train, and return
        pass

    def test_multi_output(self):
        model = self.__get_multi_output_model()
        models = [KerasClassifier((0,1), model, output_layer=i) for i in range(len(model.outputs))]
        ensemble = EnsembleClassifier((0,1), *models)
        # And now test
        pass
