# MIT License
#
# Copyright (C) IBM Corporation 2020
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

import numpy as np

from art.defences import Distillation

from tests.utils import TestBase, master_seed
from tests.utils import get_classifier_tf

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_EPOCHS = 10


class TestDistillation(TestBase):
    """
    A unittest class for testing the Distillation transformer.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

    def setUp(self):
        super().setUp()

    def test_tensorflow_classifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Create the trained classifier
        trained_classifier, sess = get_classifier_tf()

        # Create the modified classifier
        modified_classifier, _ = get_classifier_tf(
            load_init=False,
            sess=sess
        )

        # Create distillation transformer
        transformer = Distillation(
            classifier=trained_classifier,
            batch_size=BATCH_SIZE,
            nb_epochs=NB_EPOCHS
        )

        # Perform the transformation
        modified_classifier = transformer(
            x=self.x_train_mnist,
            modified_classifier=modified_classifier
        )

        # Compare the 2 outputs
        preds1 = trained_classifier.predict(
            x=self.x_train_mnist,
            batch_size=BATCH_SIZE
        )

        preds2 = modified_classifier.predict(
            x=self.x_train_mnist,
            batch_size=BATCH_SIZE
        )

        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)

        self.assertGreater(acc, 0.5)

