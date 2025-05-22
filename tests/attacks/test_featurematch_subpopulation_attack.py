# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
import unittest
import logging

import numpy as np

from art.attacks.poisoning.featurematch_subpopulation_attack import PoisoningAttackSubpopulationPoisoning

logger = logging.getLogger(__name__)

class TestFeatureMatchSubpopulationPoisoning(unittest.TestCase):

    """
    A unittest class for testing the FeatureMatch Subpopulation Poisoning attack.
    """

    def setUp(self):
        self.aux_data = np.array([[1, 1], [2, 2], [1, 1], [3, 3]])
        self.aux_labels = np.array([0, 1, 0, 2])

        self.test_data = np.array([[1, 1], [2, 2], [4, 4]])
        self.test_labels = np.array([0, 1, 2])

        self.annotations = np.array([[1, 1], [2, 2], [1, 1], [3, 3]])

        self.n_classes = 3
        self.poison_rates = [0.5, 1.0]

    def test_valid_feature_match(self):
        attack = PoisoningAttackSubpopulationPoisoning(
            aux_data=self.aux_data,
            aux_labels=self.aux_labels,
            test_data=self.test_data,
            test_labels=self.test_labels,
            n_classes=self.n_classes,
            feature_annotations=self.annotations,
            poison_rates=self.poison_rates
        )

        poison_dict = attack.feature_match()
        self.assertIsInstance(poison_dict, dict)
        self.assertGreater(len(poison_dict), 0)

        for key, value in poison_dict.items():
            self.assertIn("x_poison_samples", value)
            self.assertIn("y_poison_samples", value)
            self.assertEqual(len(value["x_poison_samples"]), value["poison count"])
            self.assertEqual(len(value["y_poison_samples"]), value["poison count"])

    def test_poison_labels_range(self):
        attack = PoisoningAttackSubpopulationPoisoning(
            aux_data=self.aux_data,
            aux_labels=self.aux_labels,
            test_data=self.test_data,
            test_labels=self.test_labels,
            n_classes=self.n_classes,
            feature_annotations=self.annotations,
            poison_rates=self.poison_rates
        )

        poison_dict = attack.feature_match()
        for value in poison_dict.values():
            self.assertTrue(np.all(value["y_poison_samples"] >= 0))
            self.assertTrue(np.all(value["y_poison_samples"] < self.n_classes))

    def test_invalid_empty_aux_data(self):
        with self.assertRaises(ValueError):
            PoisoningAttackSubpopulationPoisoning(
                aux_data=[],
                aux_labels=self.aux_labels,
                test_data=self.test_data,
                test_labels=self.test_labels,
                n_classes=self.n_classes,
                feature_annotations=self.annotations,
                poison_rates=self.poison_rates
            )

    def test_invalid_empty_aux_labels(self):
        with self.assertRaises(ValueError):
            PoisoningAttackSubpopulationPoisoning(
                aux_data=self.aux_data,
                aux_labels=[],
                test_data=self.test_data,
                test_labels=self.test_labels,
                n_classes=self.n_classes,
                feature_annotations=self.annotations,
                poison_rates=self.poison_rates
            )

    def test_invalid_annotations_none(self):
        with self.assertRaises(ValueError):
            PoisoningAttackSubpopulationPoisoning(
                aux_data=self.aux_data,
                aux_labels=self.aux_labels,
                test_data=self.test_data,
                test_labels=self.test_labels,
                n_classes=self.n_classes,
                feature_annotations=None,
                poison_rates=self.poison_rates
            )

    def test_invalid_poison_rate_zero(self):
        with self.assertRaises(ValueError):
            PoisoningAttackSubpopulationPoisoning(
                aux_data=self.aux_data,
                aux_labels=self.aux_labels,
                test_data=self.test_data,
                test_labels=self.test_labels,
                n_classes=self.n_classes,
                feature_annotations=self.annotations,
                poison_rates=[0]
            )


if __name__ == "__main__":
    unittest.main()
