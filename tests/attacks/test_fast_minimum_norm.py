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
"""Small, deterministic FMN tests; no downloaded model or dataset is required."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from art.attacks.evasion import FastMinimumNorm
from art.estimators.classification.classifier import ClassifierMixin, ClassGradientsMixin
from art.estimators.estimator import BaseEstimator
from art.exceptions import EstimatorError


class LinearClassifier(ClassGradientsMixin, ClassifierMixin, BaseEstimator):
    """Logit classifier with analytic derivatives and a known decision boundary."""

    def __init__(self, weights=None, bias=None, clip_values=(0.0, 1.0), input_shape=(2,)):
        super().__init__(model=None, clip_values=clip_values)
        self.weights = np.array([[1, 0.5], [-1, -0.5]] if weights is None else weights, dtype=np.float32)
        self.bias = np.array([-0.65, 0.65] if bias is None else bias, dtype=np.float32)
        self.nb_classes = len(self.weights)
        self._input_shape = input_shape

    @property
    def input_shape(self):
        return self._input_shape

    def predict(self, x, **kwargs):
        return x.reshape(len(x), -1) @ self.weights.T + self.bias

    def class_gradient(self, x, label=None, **kwargs):
        return self.weights[label].reshape((len(x), 1) + self.input_shape).copy()

    def fit(self, x, y, **kwargs):
        raise NotImplementedError


class TestFastMinimumNorm(unittest.TestCase):
    """Check success, minimum distance, batching, failure behavior and parameter validation."""

    def setUp(self):
        self.classifier = LinearClassifier()
        self.x = np.array([[0.1, 0.1], [0.8, 0.8], [0.2, 0.2]], dtype=np.float32)
        self.labels = self.classifier.predict(self.x).argmax(axis=1)

    def attack(self, classifier=None, **kwargs):
        return FastMinimumNorm(self.classifier if classifier is None else classifier, verbose=False, **kwargs)

    def test_near_analytic_minimum(self):
        for norm in (2, np.inf, "inf"):
            with self.subTest(norm=norm):
                adv = self.attack(norm=norm, max_iter=150, max_step_size=0.2).generate(self.x, self.labels)
                self.assertTrue(np.all(self.classifier.predict(adv).argmax(axis=1) != self.labels))
                effective_norm = 2 if norm == 2 else np.inf
                dual = 2 if norm == 2 else 1
                # Distance to w.x = 0.65 is |w.x - 0.65| / ||w||_dual; projections lie inside the box.
                minimum = abs(self.x @ np.array([1, 0.5]) - 0.65) / np.linalg.norm([1, 0.5], ord=dual)
                distance = np.linalg.norm(adv - self.x, ord=effective_norm, axis=1)
                np.testing.assert_allclose(distance, minimum, atol=0.004, rtol=0.01)
                self.assertTrue(np.all((adv >= 0) & (adv <= 1)))

    def test_labels_batching_and_no_mutation(self):
        x_original, y_original = self.x.copy(), self.labels.copy()
        adv = self.attack(batch_size=2).generate(self.x, self.labels)
        np.testing.assert_allclose(adv, self.attack(batch_size=1).generate(self.x), atol=1e-6)
        np.testing.assert_allclose(adv, self.attack().generate(self.x, np.eye(2)[self.labels]), atol=1e-6)
        np.testing.assert_allclose(adv, self.attack().generate(self.x, self.labels[:, None]), atol=1e-6)
        np.testing.assert_array_equal(self.x, x_original)
        np.testing.assert_array_equal(self.labels, y_original)
        self.assertEqual(adv.dtype, np.float32)
        self.assertEqual(self.attack().generate(np.empty((0, 2))).shape, (0, 2))

    def test_float32_linear_boundary_rounding(self):
        rng = np.random.RandomState(42)
        weight = rng.normal(size=64).astype(np.float32)
        bias = -weight @ np.full(64, 0.5, dtype=np.float32)
        classifier = LinearClassifier(weights=[weight, -weight], bias=[bias, -bias], input_shape=(64,))
        x = rng.uniform(0.25, 0.75, (8, 64)).astype(np.float32)
        labels = classifier.predict(x).argmax(axis=1)
        minimum = abs(x @ weight + bias) / np.linalg.norm(weight)
        for targeted in (False, True):
            with self.subTest(targeted=targeted):
                y = 1 - labels if targeted else labels
                adv = self.attack(classifier, targeted=targeted, max_iter=50).generate(x, y)
                np.testing.assert_array_equal(classifier.predict(adv).argmax(axis=1), 1 - labels)
                np.testing.assert_allclose(np.linalg.norm(adv - x, axis=1), minimum, atol=2e-5)

    def test_targeted_multiclass_and_image_input(self):
        classifier = LinearClassifier(weights=[[1, 0], [-1, 0], [0, 1]], bias=[-0.5, 0.5, -0.5], input_shape=(1, 2, 1))
        x = np.array([[0.8, 0.2], [0.2, 0.2]], dtype=np.float32).reshape(2, 1, 2, 1)
        targets = np.array([2, 2])
        for norm in (2, np.inf):
            with self.subTest(norm=norm):
                adv = self.attack(classifier, norm=norm, targeted=True, max_iter=100).generate(x, targets)
                np.testing.assert_array_equal(classifier.predict(adv).argmax(axis=1), targets)
                self.assertEqual(adv.shape, x.shape)
                self.assertTrue(np.all((adv >= 0) & (adv <= 1)))
        with self.assertRaisesRegex(ValueError, "Target labels"):
            self.attack(targeted=True).generate(self.x)

    def test_flat_or_unsuccessful_search_returns_original(self):
        flat = LinearClassifier(weights=np.zeros((2, 2)), bias=[1, -1])
        for norm in (2, np.inf):
            with self.subTest(norm=norm), np.errstate(all="raise"):
                np.testing.assert_array_equal(self.attack(flat, norm=norm).generate(self.x), self.x)
                np.testing.assert_array_equal(self.attack(norm=norm, max_iter=1).generate(self.x), self.x)
                # These labels already satisfy untargeted and targeted criteria respectively.
                np.testing.assert_array_equal(self.attack(norm=norm).generate(self.x, 1 - self.labels), self.x)
                np.testing.assert_array_equal(
                    self.attack(norm=norm, targeted=True).generate(self.x, self.labels), self.x
                )
                np.testing.assert_array_equal(self.attack(norm=norm, gamma=1).generate(self.x, 1 - self.labels), self.x)

    def test_featurewise_bounds_and_step_schedule(self):
        classifier = LinearClassifier(clip_values=(np.array([0, 0]), np.array([0.8, 0.9])))
        for norm in (2, np.inf):
            with self.subTest(norm=norm):
                attack = self.attack(classifier, norm=norm, min_step_size=0.01)
                adv = attack.generate(self.x)
                self.assertTrue(np.all(adv <= classifier.clip_values[1]))
                self.assertTrue(np.all(adv >= 0))
                self.assertTrue(np.any(adv != self.x))
        attack.set_params(max_iter=2, targeted=True)
        self.assertEqual(attack.max_iter, 2)
        self.assertTrue(attack.targeted)
        with self.assertRaises(ValueError):
            attack.set_params(max_iter=0)

    def test_invalid_parameters(self):
        invalid = {
            "norm": [0, 1, "L2"],
            "max_iter": [0, -1, True, 1.5],
            "batch_size": [0, -1, True, 1.5],
            "max_step_size": [0, -1, True, "1", np.inf, np.nan],
            "min_step_size": [-1, True, "0.1", 2, np.inf, np.nan],
            "gamma": [0, -1, 0.0001, 1.1, True, "0.1", np.nan],
            "targeted": [1],
            "verbose": [1],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    FastMinimumNorm(self.classifier, **{name: value})
        for bounds in (None, (-np.inf, np.inf)):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "finite estimator"):
                self.attack(LinearClassifier(clip_values=bounds))
        with self.assertRaises(EstimatorError):
            self.attack(classifier=object())

    def test_invalid_inputs_and_outputs(self):
        for x in (self.x[:, :1], np.array([0.1, 0.2]), np.array([[np.nan, 0.1]]), np.array([[-0.1, 0.2]])):
            with self.subTest(x=x), self.assertRaises(ValueError):
                self.attack().generate(x)
        for y in (
            np.array([0]),
            np.array([0, -1, 0]),
            np.array([0, 2, 0]),
            np.array([0, 0.5, 0]),
            np.array([0, np.nan, 0]),
            np.ones((3, 3)),
        ):
            with self.subTest(y=y), self.assertRaises(ValueError):
                self.attack().generate(self.x, y)
        for logits in (np.zeros((3, 1)), np.full((3, 2), np.nan)):
            with patch.object(self.classifier, "predict", return_value=logits), self.assertRaisesRegex(
                ValueError, "finite logit"
            ):
                self.attack().generate(self.x)
        with patch.object(self.classifier, "class_gradient", return_value=np.full((3, 1, 2), np.nan)):
            with self.assertRaisesRegex(ValueError, "gradients must be finite"):
                self.attack().generate(self.x)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class TestFastMinimumNormPyTorch(unittest.TestCase):
    """Exercise ART's real autograd estimator, with optional independent Foolbox parity."""

    def test_pytorch_and_optional_linf_reference(self):
        import torch
        from art.estimators.classification import PyTorchClassifier

        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
        classifier = PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(4,),
            nb_classes=3,
            clip_values=(0, 1),
            device_type="cpu",
        )
        x = np.random.RandomState(8).uniform(0.1, 0.9, (12, 4)).astype(np.float32)
        labels = classifier.predict(x).argmax(axis=1)
        for norm in (2, np.inf):
            for targeted in (False, True):
                with self.subTest(norm=norm, targeted=targeted):
                    y = (labels + 1) % 3 if targeted else labels
                    attack = FastMinimumNorm(
                        classifier, norm=norm, targeted=targeted, max_iter=50, gamma=0.05, verbose=False
                    )
                    adv = attack.generate(x, y)
                    self.assertTrue(np.all(np.isfinite(adv)))
                    self.assertTrue(np.all((adv >= 0) & (adv <= 1)))
                    if norm == np.inf and importlib.util.find_spec("foolbox"):
                        import foolbox as fb

                        reference_class = fb.attacks.L2FMNAttack if norm == 2 else fb.attacks.LInfFMNAttack
                        reference = reference_class(steps=50, max_stepsize=1.0, min_stepsize=0.01, gamma=0.05)
                        criterion = fb.criteria.TargetedMisclassification if targeted else fb.criteria.Misclassification
                        reference_adv = reference.run(
                            fb.PyTorchModel(model, bounds=(0, 1), device="cpu"),
                            torch.from_numpy(x),
                            criterion(torch.from_numpy(y)),
                        )
                        np.testing.assert_allclose(adv, reference_adv.numpy(), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
