# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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

from art.attacks.evasion import PatchFoolPyTorch
from art.estimators.classification.classifier import ClassGradientsMixin
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.estimator import BaseEstimator

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

@pytest.fixture()
def get_pytorch_deit(get_default_cifar10_subset):

    import cv2
    import torch
    from art.estimators.classification import PyTorchClassifier

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    model = torch.hub.load('facebookresearch/deit', 'deit_base_patch16_224', pretrained=True)
    patch_size = 16

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(MEAN, STD),
    )
    
    return_nodes = [
            "_model.blocks.0.attn.softmax",
            "_model.blocks.1.attn.softmax",
            "_model.blocks.2.attn.softmax",
            "_model.blocks.3.attn.softmax",
            "_model.blocks.4.attn.softmax",
            "_model.blocks.5.attn.softmax",
            "_model.blocks.6.attn.softmax",
            "_model.blocks.7.attn.softmax",
            "_model.blocks.8.attn.softmax",
            "_model.blocks.9.attn.softmax",
            "_model.blocks.10.attn.softmax",
            "_model.blocks.11.attn.softmax",
        ]

    (_, _), (x_test_cifar10, y_test_cifar10) = get_default_cifar10_subset

    x_test = cv2.resize(
    x_test_cifar10[0].transpose((1, 2, 0)), dsize=(224, 224), interpolation=cv2.INTER_CUBIC
    ).transpose((2, 0, 1))
    x_test = np.expand_dims(x_test, axis=0)
    x_test = np.repeat(x_test, repeats=2, axis=0)

    return classifier, return_nodes, patch_size, x_test, y_test_cifar10

@pytest.mark.skip_framework("tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_generate_no_labels(art_warning, get_pytorch_deit):
    try:

        classifier, return_nodes, patch_size, x_test, _ = get_pytorch_deit

        attack = PatchFoolPyTorch(estimator=classifier, 
                        attention_nodes=return_nodes, 
                        patch_size=patch_size, 
                        max_iter=10, 
                        random_start=False)
                        
        x_test_adv = attack.generate(x=x_test)

        assert np.mean(x_test) == pytest.approx(0.4250448, 0.01)
        assert np.mean(np.abs(x_test_adv - x_test)) != 0.0

    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework("tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def test_check_params(art_warning, get_pytorch_deit):
    try:
        classifier, return_nodes, patch_size, _, _ = get_pytorch_deit

        with pytest.raises(TypeError):
            _ = PatchFoolPyTorch(classifier, attention_nodes=0, patch_size=patch_size)

        with pytest.raises(ValueError):
            _ = PatchFoolPyTorch(classifier, attention_nodes=return_nodes, patch_size=-1)

        with pytest.raises(ValueError):
            _ = PatchFoolPyTorch(classifier, attention_nodes=return_nodes, patch_size=patch_size, max_iter=-1)

        with pytest.raises(ValueError):
            _ = PatchFoolPyTorch(classifier, attention_nodes=return_nodes, patch_size=patch_size, batch_size=0)

        with pytest.raises(ValueError):
            _ = PatchFoolPyTorch(classifier, attention_nodes=return_nodes, patch_size=patch_size, patch_layer=-1)

    except ARTTestException as e:
        art_warning(e)
