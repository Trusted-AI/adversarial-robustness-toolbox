from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Conv1D, LeakyReLU, MaxPooling1D, Dense, Flatten
from keras.models import Sequential

from art.attacks.configurable_text_attack import ConfigurableTextAttack, check_prediction_change

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 1
EMB_SIZE = 32
MAX_LENGTH = 500


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv = nn.Conv2d(1, 16, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc = nn.Linear(2304, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv(x)))
#         x = x.view(-1, 2304)
#         logit_output = self.fc(x)
#
#         return logit_output


# class PtFlatten(nn.Module):
#     def forward(self, x):
#         n = x.size()
#         result = x.view(n[0], -1)


class TestConfigurableTextAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from art.utils import load_imdb, to_categorical
        from art.classifiers import KerasTextClassifier, TFTextClassifier

        k.clear_session()
        k.set_learning_phase(1)

        # Load IMDB
        (x_train, y_train), (x_test, y_test), ids = load_imdb(nb_words=1000, max_length=MAX_LENGTH)
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.imdb = (x_train, y_train), (x_test, y_test)

        ids = [value for key, value in ids.items()]
        cls.word_ids = ids

        # Create basic Keras word model on IMDB
        model = Sequential()
        model.add(Embedding(1000, EMB_SIZE, input_length=MAX_LENGTH))
        model.add(Conv1D(filters=16, kernel_size=3))
        model.add(LeakyReLU(alpha=.2))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(units=100))
        model.add(LeakyReLU(alpha=.2))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile, fit and store model in class
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=2, batch_size=BATCH_SIZE)
        cls.classifier_k = KerasTextClassifier(model=model, ids=cls.word_ids)

        scores = cls.classifier_k._model.evaluate(x_test, y_test)
        logger.info('[Keras, IMDB] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # Create basic TF model on IMDB
        input_ph = tf.placeholder(tf.int32, shape=[None, 500])
        output_ph = tf.placeholder(tf.int32, shape=[None, 2])

        # Define the TF graph
        embedding_layer = tf.keras.layers.Embedding(1000, 32, input_length=500)(input_ph)
        conv_1d = tf.keras.layers.Conv1D(filters=16, kernel_size=3)(embedding_layer)
        lkrelu1 = tf.keras.layers.LeakyReLU(alpha=.2)(conv_1d)
        mp = tf.keras.layers.MaxPool1D()(lkrelu1)
        flatten = tf.layers.flatten(mp)
        dense = tf.layers.dense(flatten, units=100)
        lkrelu2 = tf.keras.layers.LeakyReLU(alpha=.2)(dense)
        logits = tf.layers.dense(lkrelu2, units=2)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        cls.sess = tf.Session()
        cls.sess.run(tf.global_variables_initializer())

        # Create classifier
        cls.classifier_tf = TFTextClassifier(input_ph, logits, embedding_layer, ids, output_ph, train, loss, None,
                                             cls.sess)
        cls.classifier_tf.fit(x_train, to_categorical(y_train), nb_epochs=2, batch_size=BATCH_SIZE)

        scores = np.argmax(cls.classifier_tf.predict(x_test), axis=1)
        acc = np.sum(scores == y_test) / y_test.shape[0]
        logger.info('[TF, IMDB] Accuracy on test set: %.2f%%', (acc * 100))

        # PyTorch classifier
        # model = nn.Sequential(nn.Embedding(1000, 32, sparse=False), PtFlatten(), nn.Linear(16000, 8), nn.Linear(8, 2))
        # loss_fn = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        #
        # # Create classifier
        # cls.classifier_pt = PyTorchTextClassifier(model, 0, ids, loss_fn, optimizer, 2)
        # cls.classifier_pt.fit(x_train.astype(np.int64), to_categorical(y_train, nb_classes=2), nb_epochs=2,
        #                       batch_size=BATCH_SIZE)

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()
        tf.reset_default_graph()
        k.clear_session()

    def test_imdb_fgsm(self):
        from art.attacks.configurable_text_attack import TextFGSM, loss_gradient_score

        (_, _), (x_test, y_test) = self.imdb
        models = {'Keras': self.classifier_k, 'TF': self.classifier_tf}
        for backend, model in models.items():
            logger.info('Text FGSM attack on %s', backend)
            attack = ConfigurableTextAttack(classifier=model, nb_changes=2, transform=TextFGSM(eps=200),
                                            score=loss_gradient_score, stop_condition=check_prediction_change)
            x_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_adv).all())

    def test_imdb_head(self):
        from art.attacks.configurable_text_attack import TextFGSM, TemporalHeadScore

        (_, _), (x_test, y_test) = self.imdb
        models = {'Keras': self.classifier_k, 'TF': self.classifier_tf}
        scorer = TemporalHeadScore()

        for backend, model in models.items():
            logger.info('Text head score attack on %s', backend)

            attack = ConfigurableTextAttack(classifier=model, nb_changes=2, score=scorer, transform=TextFGSM(eps=200),
                                            stop_condition=check_prediction_change)
            x_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_adv).all())

    def test_imdb_combined(self):
        from art.attacks.configurable_text_attack import CombinedScore, TextFGSM

        (_, _), (x_test, y_test) = self.imdb
        models = {'Keras': self.classifier_k, 'TF': self.classifier_tf}
        scorer = CombinedScore()

        for backend, model in models.items():
            logger.info('Text combined score attack on %s', backend)
            attack = ConfigurableTextAttack(classifier=model, nb_changes=2, score=scorer, transform=TextFGSM(eps=1000),
                                            stop_condition=check_prediction_change)
            x_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_adv).all())


class TestFunctions(unittest.TestCase):
    def test_pred_changed(self):
        class DummyClassifier:
            iteration = 0

            @property
            def nb_classes(self):
                return 10

            def predict(self, x):
                preds = np.zeros((x.shape[0], self.nb_classes))
                if not self.iteration % 2:
                    preds[:, 0] = 0.5
                else:
                    preds[:, 3] = 0.5
                self.iteration += 1

                return preds

        classifier = DummyClassifier()
        x1 = np.zeros((28, 28, 1))
        x2 = np.zeros((28, 28, 1))
        cond = check_prediction_change(classifier, x1, x2)
        self.assertTrue(cond)

    def test_pred_unchanged(self):
        class DummyClassifier:
            @property
            def nb_classes(self):
                return 10

            def predict(self, x):
                preds = np.zeros((x.shape[0], self.nb_classes))
                preds[:, 3] = 0.5

                return preds

        classifier = DummyClassifier()
        x1 = np.zeros((28, 28, 1))
        x2 = np.zeros((28, 28, 1))
        cond = check_prediction_change(classifier, x1, x2)
        self.assertFalse(cond)

    def test_head_tail_scores(self):
        from art.attacks.configurable_text_attack import TemporalHeadScore, TemporalTailScore, CombinedScore

        class DummyClassifier:
            @property
            def nb_classes(self):
                return 10

            def predict(self, x):
                return np.ones((x.shape[0], self.nb_classes))

        classifier = DummyClassifier()
        x = np.ones(11)

        # Test head score
        ths = TemporalHeadScore()
        scores = ths(classifier, x, None)
        self.assertTrue(scores.shape == x.shape)

        # Check expected head score
        expected = np.zeros(11)
        expected[0] = 1
        self.assertTrue((scores == expected).all())

        # Test tail score
        tts = TemporalTailScore()
        scores = tts(classifier, x, None)
        self.assertTrue(scores.shape == x.shape)

        # Check expected tail score
        expected[0] = 0
        expected[-1] = 1
        self.assertTrue((scores == expected).all())

        # Test combined score
        cs = CombinedScore()
        scores = cs(classifier, x, None)
        self.assertTrue(scores.shape == x.shape)

        # Check expected combined score
        expected[0] = 1
        self.assertTrue((scores == expected).all())


class TestTextFGSM(unittest.TestCase):
    def test_gradients(self):
        class DummyClassifier:
            def to_embedding(self, x):
                return np.ones((x.shape[0], x.shape[1], EMB_SIZE))

            def loss_gradient(self, x, y):
                return np.reshape(np.arange(np.prod(x.shape) * EMB_SIZE), x.shape + (EMB_SIZE,))

        from art.attacks.configurable_text_attack import TextFGSM
        transform = TextFGSM(eps=10)
        classifier = DummyClassifier()
        x = np.zeros((28, 28, 1))
        y = np.array([1, 0])

        self.assertTrue(transform.uses_embedding)

        res = transform(classifier, x, y)
        self.assertTrue((np.diff(res) > 0).all())

        transform2 = TextFGSM(eps=20)
        res2 = transform2(classifier, x, y)
        x_emb = classifier.to_embedding(x)
        self.assertTrue((res2 - x_emb == 2 * (res - x_emb)).all())


if __name__ == '__main__':
    unittest.main()
