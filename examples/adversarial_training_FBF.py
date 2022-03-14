"""
This is an example of how to use ART for adversarial training of a model with Fast is better than free protocol
"""
import math
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from art.classifiers import PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.defences.trainer import AdversarialTrainerFBFPyTorch
from art.utils import load_cifar10
from art.attacks.evasion import ProjectedGradientDescent

"""
For this example we choose the PreActResNet model as used in the paper (https://openreview.net/forum?id=BJx040EFvH)
The code for the model architecture has been adopted from
https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_CIFAR10/blob/master/preact_resnet.py
"""


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2.0 / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class CIFAR10_dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = Image.fromarray(((self.data[index] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
        x = self.transform(x)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


# Step 1: Load the CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

cifar_mu = np.ones((3, 32, 32))
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465

# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

cifar_std = np.ones((3, 32, 32))
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616

x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
)

dataset = CIFAR10_dataset(x_train, y_train, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Step 2: create the PyTorch model
model = PreActResNet18()
# For running on GPU replace the model with the
# model = PreActResNet18().cuda()

model.apply(initialize_weights)
model.train()

opt = torch.optim.SGD(model.parameters(), lr=0.21, momentum=0.9, weight_decay=5e-4)

# if you have apex installed, the following line should be uncommented for faster processing
# import apex.amp as amp
# model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

criterion = nn.CrossEntropyLoss()
# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    preprocessing=(cifar_mu, cifar_std),
    loss=criterion,
    optimizer=opt,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

attack = ProjectedGradientDescent(
    classifier,
    norm=np.inf,
    eps=8.0 / 255.0,
    eps_step=2.0 / 255.0,
    max_iter=40,
    targeted=False,
    num_random_init=5,
    batch_size=32,
)

# Step 4: Create the trainer object - AdversarialTrainerFBFPyTorch
# if you have apex installed, change use_amp to True
epsilon = 8.0 / 255.0
trainer = AdversarialTrainerFBFPyTorch(classifier, eps=epsilon, use_amp=False)

# Build a Keras image augmentation object and wrap it in ART
art_datagen = PyTorchDataGenerator(iterator=dataloader, size=x_train.shape[0], batch_size=128)

# Step 5: fit the trainer
trainer.fit_generator(art_datagen, nb_epochs=30)

x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
print(
    "Accuracy on benign test samples after adversarial training: %.2f%%"
    % (np.sum(x_test_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
)

x_test_attack = attack.generate(x_test)
x_test_attack_pred = np.argmax(classifier.predict(x_test_attack), axis=1)
print(
    "Accuracy on original PGD adversarial samples after adversarial training: %.2f%%"
    % (np.sum(x_test_attack_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
)
