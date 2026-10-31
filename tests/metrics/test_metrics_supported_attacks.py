# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2026
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
"""Fast, CPU-only regression tests for the metrics' attack configuration."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from art.attacks.evasion import AutoAttack, FastGradientMethod
from art.estimators.classification import SklearnClassifier
from art.metrics import adversarial_accuracy, empirical_robustness


class TestMetricsSupportedAttacks(unittest.TestCase):
    def setUp(self):
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)
        np.random.seed(1234)
        self.x = np.array([[0.1, 0.1], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        self.classifier = SklearnClassifier(LogisticRegression(C=100).fit(self.x, y), clip_values=(0.0, 1.0))

    def test_adversarial_accuracy_with_hop_skip_jump(self):
        accuracy = adversarial_accuracy(
            self.classifier,
            self.x,
            attack_name="hsj",
            attack_params={"max_iter": 1, "max_eval": 10, "init_eval": 10, "init_size": 10, "verbose": False},
        )
        self.assertEqual(accuracy, 0.0)

    def test_empirical_robustness_with_hop_skip_jump(self):
        robustness = empirical_robustness(
            self.classifier,
            self.x,
            attack_name="hsj",
            attack_params={"max_iter": 1, "max_eval": 10, "init_eval": 10, "init_size": 10, "verbose": False},
        )
        self.assertTrue(np.isfinite(robustness))
        self.assertGreater(robustness, 0.0)

    def test_metrics_with_auto_attack(self):
        for metric in (adversarial_accuracy, empirical_robustness):
            with self.subTest(metric=metric.__name__):
                # Use a short attack list so this real AutoAttack needs no deep-learning framework.
                attack = AutoAttack(
                    self.classifier,
                    eps=1.0,
                    attacks=[FastGradientMethod(self.classifier, eps=1.0)],
                )
                with patch("art.metrics.metrics.get_crafter", return_value=attack):
                    result = metric(self.classifier, self.x, attack_name="auto")
                self.assertTrue(np.isfinite(result))
                if metric is adversarial_accuracy:
                    self.assertEqual(result, 0.0)
                else:
                    self.assertGreater(result, 0.0)

    def test_metrics_still_enable_fgsm_minimal_mode(self):
        for metric in (adversarial_accuracy, empirical_robustness):
            with self.subTest(metric=metric.__name__):
                attack = FastGradientMethod(self.classifier, eps=1.0, eps_step=0.1, minimal=False)
                with patch("art.metrics.metrics.get_crafter", return_value=attack):
                    result = metric(self.classifier, self.x, attack_name="fgsm")
                self.assertTrue(attack.minimal)
                self.assertTrue(np.isfinite(result))

    def test_supplied_crafter_keeps_its_minimal_setting(self):
        attack = FastGradientMethod(self.classifier, eps=1.0, minimal=False)
        accuracy = adversarial_accuracy(self.classifier, self.x, attack_crafter=attack)
        self.assertEqual(accuracy, 0.0)
        self.assertFalse(attack.minimal)


if __name__ == "__main__":
    unittest.main()
