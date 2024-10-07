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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest
import numpy as np

import sys
import os 

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.defences.detector.evasion import BeyondDetector
from art.utils import load_dataset, get_file


from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

import torch.nn as nn
from torchvision import models, transforms

class SimSiamWithCls(nn.Module):
    '''
    SimSiam with Classifier
    '''
    def __init__(self, arch='resnet18', feat_dim=2048, num_proj_layers=2):
        
        super(SimSiamWithCls, self).__init__()
        self.backbone = models.resnet18()
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(out_dim, 10)

        pred_hidden_dim = int(feat_dim / 4)

        self.projector = nn.Sequential(
            nn.Linear(out_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, feat_dim),
        )

    def forward(self, img, im_aug1=None, im_aug2=None):

        r_ori = self.backbone(img)
        if im_aug1 is None and im_aug2 is None:
            cls = self.classifier(r_ori)
            rep = self.projector(r_ori)
            return {'cls': cls, 'rep':rep}
        else:

            r1 = self.backbone(im_aug1)
            r2 = self.backbone(im_aug2)

            z1 = self.projector(r1)
            z2 = self.projector(r2)
            # print("shape of z:", z1.shape)

            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            # print("shape of p:", p1.shape)

            return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}

@pytest.fixture
def get_cifar10():
    """
    Loads CIFAR10 dataset.
    """
    (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
    return (x_train, y_train), (x_test, y_test), min_, max_


@pytest.fixture
def get_ssl_model(weights_path):
    """
    Loads the SSL model (SimSiamWithCls).
    """
    model = SimSiamWithCls()
    model.load_state_dict(torch.load(weights_path))
    return model

@pytest.mark.only_with_platform("pytorch")
def test_beyond_detector(art_warning, get_cifar10, get_ssl_model):
    try:
        # Load CIFAR10 data
        (x_train, y_train), (x_test, y_test), min_, max_ = get_cifar10

        # Load models
        # Download pretrained weights from https://drive.google.com/drive/folders/1ieEdd7hOj2CIl1FQfu4-3RGZmEj-mesi?usp=sharing
        target_model = models.resnet18()
        target_model.load_state_dict(torch.load("./resnet_c10.pth"))
        ssl_model = get_ssl_model()
        ssl_model.load_state_dict(torch.load("./simsiam_c10.pth"))


        # Generate adversarial samples
        attack = FastGradientMethod(estimator=target_model, eps=0.05)
        x_test_adv = attack.generate(x_test)

        img_augmentations = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Initialize BeyondDetector
        detector = BeyondDetector(
            target_model=target_model,
            ssl_model=ssl_model,
            img_augmentation=img_augmentations,
            aug_num=50,
            alpha=0.8,
            K=20,
            percentile=5
        )

        # Fit the detector
        detector.fit(x_train, y_train, batch_size=128)

        # Apply detector on clean and adversarial test data
        _, test_detection = detector.detect(x_test)
        _, test_adv_detection = detector.detect(x_test_adv)

        # Assert there is at least one true positive and negative
        nb_true_positives = np.sum(test_adv_detection)
        nb_true_negatives = len(test_detection) - np.sum(test_detection)
        assert nb_true_positives > 0
        assert nb_true_negatives > 0

        # Calculate and print detection accuracy
        clean_accuracy = 1 - np.mean(test_detection)
        adv_accuracy = np.mean(test_adv_detection)
        print(f"Clean Detection Accuracy: {clean_accuracy:.4f}")
        print(f"Adversarial Detection Accuracy: {adv_accuracy:.4f}")

    except ARTTestException as e:
        art_warning(e)

if __name__ == "__main__":

    test_beyond_detector()