# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE for any claim, damages or other liability, whether in an action of contract,
# TORT OR OTHERWISE, ARISING from, out of or in connection with the software or the use or other dealings in the
# Software.

import logging

import numpy as np
import pytest

from art.attacks.poisoning.one_pixel_shortcut_attack import OnePixelShortcutAttack
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

def test_one_pixel_per_image_and_label_preservation():
    try:
        x = np.zeros((4, 3, 3))
        y = np.array([0, 0, 1, 1])
        attack = OnePixelShortcutAttack()
        x_p, y_p = attack.poison(x.copy(), y.copy())

        assert x_p.shape == x.shape
        assert np.array_equal(y_p, y)

        changes = np.sum(x_p != x, axis=(1, 2))
        assert np.all(changes == 1)

        coords = [tuple(np.argwhere(x_p[i] != x[i])[0]) for i in range(x.shape[0])]
        assert coords[0] == coords[1]
        assert coords[2] == coords[3]
    except Exception as e:
        logger.warning("test_one_pixel_per_image_and_label_preservation failed: %s", e)
        raise ARTTestException("Pixel change or label consistency check failed") from e

def test_missing_labels_raises_error():
    try:
        x = np.zeros((3, 5, 5))
        with pytest.raises(ValueError):
            OnePixelShortcutAttack().poison(x.copy(), None)
    except Exception as e:
        logger.warning("test_missing_labels_raises_error failed: %s", e)
        raise ARTTestException("Expected error not raised for missing labels") from e

def test_multi_channel_consistency():
    try:
        x = np.zeros((2, 2, 2, 3))
        y = np.array([0, 1])
        attack = OnePixelShortcutAttack()
        x_p, y_p = attack.poison(x.copy(), y.copy())

        assert x_p.shape == x.shape
        assert np.array_equal(y_p, y)

        diff_any = np.any(x_p != x, axis=3)
        changes = np.sum(diff_any, axis=(1, 2))
        assert np.all(changes == 1)

        coords0 = np.argwhere(diff_any[0])
        coords1 = np.argwhere(diff_any[1])
        assert coords0.shape[0] == 1
        assert coords1.shape[0] == 1
    except Exception as e:
        logger.warning("test_multi_channel_consistency failed: %s", e)
        raise ARTTestException("Multi-channel image consistency check failed") from e

def test_one_pixel_effect_with_pytorchclassifier():
    try:
        import torch
        import torch.nn as nn
        from art.estimators.classification import PyTorchClassifier

        torch.manual_seed(0)
        np.random.seed(0)

        # Create a toy dataset: 2x2 grayscale images, 2 classes
        X = np.zeros((8, 1, 2, 2), dtype=np.float32)
        for i in range(4):
            X[i, 0, 0, 0] = i * 0.25  # class 0
        for i in range(4, 8):
            X[i, 0, 0, 1] = (i - 4) * 0.25  # class 1
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        model_clean = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        loss_fn = nn.CrossEntropyLoss()

        classifier_clean = PyTorchClassifier(
            model=model_clean,
            loss=loss_fn,
            optimizer=torch.optim.SGD(model_clean.parameters(), lr=0.1),
            input_shape=(1, 2, 2),
            nb_classes=2,
        )
        classifier_clean.fit(X, y, nb_epochs=10, batch_size=4, verbose=0)
        preds_clean = classifier_clean.predict(X)
        acc_clean = np.mean(preds_clean.argmax(axis=1) == y)

        ops_attack = OnePixelShortcutAttack()
        X_poison, y_poison = ops_attack.poison(X.copy(), y.copy())

        model_poisoned = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        classifier_poisoned = PyTorchClassifier(
            model=model_poisoned,
            loss=loss_fn,
            optimizer=torch.optim.SGD(model_poisoned.parameters(), lr=0.1),
            input_shape=(1, 2, 2),
            nb_classes=2,
        )
        classifier_poisoned.fit(
            X_poison,
            y_poison,
            nb_epochs=10,
            batch_size=4,
            verbose=0,
        )
        preds_poisoned = classifier_poisoned.predict(X_poison)
        acc_poisoned = np.mean(preds_poisoned.argmax(axis=1) == y_poison)

        # Adjusted assertions for robustness
        assert acc_poisoned >= 1.0, (
            f"Expected 100% poisoned accuracy, got {acc_poisoned:.3f}"
        )
        assert acc_clean < 0.95, f"Expected clean accuracy < 95%, got {acc_clean:.3f}"

    except Exception as e:
        logger.warning("test_one_pixel_effect_with_pytorchclassifier failed: %s", e)
        raise ARTTestException(
            "PyTorchClassifier integration with OPS attack failed"
        ) from e
