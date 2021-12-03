# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import os
import logging

import numpy as np
import pytest

from art.attacks.evasion import AdversarialTexturePyTorch
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_tracking.object_tracker import ObjectTrackerMixin
from art.estimators.object_tracking import PyTorchGoturn

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_module("scripts")
@pytest.mark.skip_framework("tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_generate(art_warning, fix_get_mnist_subset, fix_get_goturn, framework):
    try:
        import torch
        from scripts.train import GoturnTrain
        from pathlib import Path

        _device = "cpu"

        goturn_path = os.path.join(os.sep, "tmp", "goturn-pytorch")

        model_dir = Path(os.path.join(goturn_path, "src", "goturn", "models"))
        ckpt_dir = model_dir.joinpath("checkpoints")
        ckpt_path = next(ckpt_dir.glob("*.ckpt"))

        ckpt_mod = torch.load(
            os.path.join(goturn_path, "src", "goturn", "models", "checkpoints", "_ckpt_epoch_3.ckpt"),
            map_location=_device,
        )
        ckpt_mod["hparams"]["pretrained_model"] = os.path.join(
            goturn_path, "src", "goturn", "models", "pretrained", "caffenet_weights.npy"
        )
        torch.save(ckpt_mod, os.path.join(goturn_path, "src", "goturn", "models", "checkpoints", "_ckpt_epoch_3.ckpt"))

        model = GoturnTrain.load_from_checkpoint(ckpt_path)

        pgt = PyTorchGoturn(
            model=model,
            input_shape=(3, 227, 227),
            clip_values=(0, 255),
            preprocessing=(np.array([104.0, 117.0, 123.0]), np.array([1.0, 1.0, 1.0])),
            device_type=_device,
        )

        y_init = np.array([[48, 79, 80, 110], [48, 79, 80, 110]])
        x_list = list()
        for i in range(2):
            x_list.append(np.random.random_integers(0, 255, size=(4, 277, 277, 3)).astype(float) / 255.0)

        x = np.asarray(x_list, dtype=float)

        y_pred = pgt.predict(x=x, y_init=y_init)

        attack = AdversarialTexturePyTorch(
            pgt,
            patch_height=4,
            patch_width=4,
            x_min=2,
            y_min=2,
            step_size=1.0 / 255.0,
            max_iter=5,
            batch_size=16,
            verbose=True,
        )

        patch = attack.generate(x=x, y=y_pred, y_init=y_init)
        assert patch.shape == (2, 4, 277, 277, 3)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_apply_patch(art_warning, fix_get_goturn):
    try:
        goturn = fix_get_goturn
        attack = AdversarialTexturePyTorch(
            goturn,
            patch_height=4,
            patch_width=4,
            x_min=2,
            y_min=2,
            step_size=1.0 / 255.0,
            max_iter=500,
            batch_size=16,
            verbose=True,
        )

        patch = np.ones(shape=(4, 4, 3))
        foreground = np.ones(shape=(1, 15, 10, 10, 3))
        foreground[:, :, 5, 5, :] = 0
        x = np.zeros(shape=(1, 15, 10, 10, 3))

        patched_images = attack.apply_patch(x=x, patch_external=patch, foreground=foreground)

        patch_sum_expected = 15 * 3 * (4 * 4 - 1)
        complement_sum_expected = 0.0

        patch_sum = np.sum(patched_images[0, :, 2:6, 2:6, :])
        complement_sum = np.sum(patched_images[0]) - patch_sum

        assert patch_sum == patch_sum_expected
        assert complement_sum == complement_sum_expected

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_check_params(art_warning, fix_get_goturn):
    try:
        goturn = fix_get_goturn

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=-2, patch_width=2)
        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2.0, patch_width=2)

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=-2)
        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2.0)

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, x_min=-2)
        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, x_min=2.0)

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, y_min=-2)
        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, y_min=2.0)

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, step_size=-2.0)
        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, step_size=2)

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, max_iter=-2)
        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, max_iter=2.0)

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, batch_size=-2)
        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, batch_size=2.0)

        with pytest.raises(ValueError):
            _ = AdversarialTexturePyTorch(goturn, patch_height=2, patch_width=2, verbose="true")

    except ARTTestException as e:
        art_warning(e)


# @pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(
            AdversarialTexturePyTorch,
            [BaseEstimator, LossGradientsMixin, ObjectTrackerMixin],
            patch_height=2,
            patch_width=2,
        )
    except ARTTestException as e:
        art_warning(e)
