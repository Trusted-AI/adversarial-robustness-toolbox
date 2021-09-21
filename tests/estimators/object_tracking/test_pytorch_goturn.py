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
import logging

import numpy as np
import pytest

from art import config
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("tensorflow", "tensorflow2v1", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_pytorch_goturn(art_warning, get_mnist_dataset):
    try:
        from art.estimators.object_tracking import PyTorchGoturn

        import os
        import torch
        from scripts.train import GoturnTrain
        from pathlib import Path

        _device = 'cpu'

        goturn_path = os.path.join(config.ART_DATA_PATH, "goturn-pytorch")

        model_dir = Path(os.path.join(goturn_path, "src", "goturn", "models"))
        ckpt_dir = model_dir.joinpath("checkpoints")
        ckpt_path = next(ckpt_dir.glob("*.ckpt"))

        ckpt_mod = torch.load(os.path.join(
            goturn_path, "src", "goturn", "models", "checkpoints", "_ckpt_epoch_3.ckpt"), map_location=_device
        )
        ckpt_mod["hparams"]["pretrained_model"] = os.path.join(
            goturn_path, "src", "goturn", "models", "pretrained", "caffenet_weights.npy"
        )
        torch.save(ckpt_mod, os.path.join(
            goturn_path, "src", "goturn", "models", "checkpoints", "_ckpt_epoch_3.ckpt"))

        model = GoturnTrain.load_from_checkpoint(ckpt_path)

        pgt = PyTorchGoturn(model=model, clip_values=(0, 255),
                            preprocessing=(np.array([104.0, 117.0, 123.0]), np.array([1.0, 1.0, 1.0])),
                            device_type=_device)

    except ARTTestException as e:
        art_warning(e)
