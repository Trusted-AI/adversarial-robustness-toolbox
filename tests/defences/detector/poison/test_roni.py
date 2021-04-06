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

import numpy as np
from sklearn.svm import SVC

from art.attacks.poisoning.poisoning_attack_svm import PoisoningAttackSVM
from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnSVC
from art.defences.detector.poison.roni import RONIDefense
from art.utils import load_mnist

from tests.utils import master_seed

logger = logging.getLogger(__name__)

NB_TRAIN, NB_POISON, NB_VALID, NB_TRUSTED = 40, 5, 40, 15
kernel = "linear"


class TestRONI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        zero_or_four = np.logical_or(y_train == 4, y_train == 0)
        x_train = x_train[zero_or_four]
        y_train = y_train[zero_or_four]
        tr_labels = np.zeros((y_train.shape[0], 2))
        tr_labels[y_train == 0] = np.array([1, 0])
        tr_labels[y_train == 4] = np.array([0, 1])
        y_train = tr_labels

        zero_or_four = np.logical_or(y_test == 4, y_test == 0)
        x_test = x_test[zero_or_four]
        y_test = y_test[zero_or_four]
        te_labels = np.zeros((y_test.shape[0], 2))
        te_labels[y_test == 0] = np.array([1, 0])
        te_labels[y_test == 4] = np.array([0, 1])
        y_test = te_labels

        n_samples_train = x_train.shape[0]
        n_features_train = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
        n_samples_test = x_test.shape[0]
        n_features_test = x_test.shape[1] * x_test.shape[2] * x_test.shape[3]

        x_train = x_train.reshape(n_samples_train, n_features_train)
        x_test = x_test.reshape(n_samples_test, n_features_test)
        x_train = x_train[:NB_TRAIN]
        y_train = y_train[:NB_TRAIN]

        trusted_data = x_test[:NB_TRUSTED]
        trusted_labels = y_test[:NB_TRUSTED]
        x_test = x_test[NB_TRUSTED:]
        y_test = y_test[NB_TRUSTED:]
        valid_data = x_test[:NB_VALID]
        valid_labels = y_test[:NB_VALID]
        x_test = x_test[NB_VALID:]
        y_test = y_test[NB_VALID:]

        no_defense = ScikitlearnSVC(model=SVC(kernel=kernel, gamma="auto"), clip_values=(min_, max_))
        no_defense.fit(x=x_train, y=y_train)
        poison_points = np.random.randint(no_defense._model.support_vectors_.shape[0], size=NB_POISON)
        all_poison_init = np.copy(no_defense._model.support_vectors_[poison_points])
        poison_labels = np.array([1, 1]) - no_defense.predict(all_poison_init)

        svm_attack = PoisoningAttackSVM(
            classifier=no_defense,
            x_train=x_train,
            y_train=y_train,
            step=0.1,
            eps=1.0,
            x_val=valid_data,
            y_val=valid_labels,
            max_iter=200,
        )

        poisoned_data, _ = svm_attack.poison(all_poison_init, y=poison_labels)

        # Stack on poison to data and add provenance of bad actor
        all_data = np.vstack([x_train, poisoned_data])
        all_labels = np.vstack([y_train, poison_labels])

        model = SVC(kernel=kernel, gamma="auto")
        cls.mnist = (
            (all_data, all_labels),
            (x_test, y_test),
            (trusted_data, trusted_labels),
            (valid_data, valid_labels),
            (min_, max_),
        )
        cls.classifier = SklearnClassifier(model=model, clip_values=(min_, max_))

        cls.classifier.fit(all_data, all_labels)
        cls.defense_cal = RONIDefense(
            cls.classifier,
            all_data,
            all_labels,
            trusted_data,
            trusted_labels,
            eps=0.1,
            calibrated=True,
        )
        cls.defence_no_cal = RONIDefense(
            cls.classifier,
            all_data,
            all_labels,
            trusted_data,
            trusted_labels,
            eps=0.1,
            calibrated=False,
        )

    def setUp(self):
        master_seed(seed=1234)

    def test_wrong_parameters_1(self):
        self.assertRaises(ValueError, self.defence_no_cal.set_params, eps=-2.0)
        self.assertRaises(ValueError, self.defense_cal.set_params, eps=-2.0)

    def test_wrong_parameters_2(self):
        (all_data, _), (_, y_test), (_, _), (_, _), (_, _) = self.mnist
        self.assertRaises(
            ValueError,
            self.defence_no_cal.set_params,
            x_train=-all_data,
            y_train=y_test,
        )
        self.assertRaises(ValueError, self.defense_cal.set_params, x_train=-all_data, y_train=y_test)

    def test_detect_poison(self):
        _, clean_trust = self.defense_cal.detect_poison()
        _, clean_no_trust = self.defence_no_cal.detect_poison()
        real_clean = np.array([1 if i < NB_TRAIN else 0 for i in range(NB_TRAIN + NB_POISON)])
        pc_tp_cal = np.average(real_clean[:NB_TRAIN] == clean_trust[:NB_TRAIN])
        pc_tn_cal = np.average(real_clean[NB_TRAIN:] == clean_trust[NB_TRAIN:])
        self.assertGreaterEqual(pc_tn_cal, 0)
        self.assertGreaterEqual(pc_tp_cal, 0.7)

        pc_tp_no_cal = np.average(real_clean[:NB_TRAIN] == clean_no_trust[:NB_TRAIN])
        pc_tn_no_cal = np.average(real_clean[NB_TRAIN:] == clean_no_trust[NB_TRAIN:])
        self.assertGreaterEqual(pc_tn_no_cal, 0)
        self.assertGreaterEqual(pc_tp_no_cal, 0.7)

    def test_evaluate_defense(self):
        real_clean = np.array([1 if i < NB_TRAIN else 0 for i in range(NB_TRAIN + NB_POISON)])
        self.defence_no_cal.detect_poison()
        self.defense_cal.detect_poison()
        logger.info(self.defense_cal.evaluate_defence(real_clean))
        logger.info(self.defence_no_cal.evaluate_defence(real_clean))


if __name__ == "__main__":
    unittest.main()
