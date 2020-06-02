"""
This is an example of how to use ART and Keras to perform adversarial training using data generators for CIFAR10
"""
from keras.preprocessing.image import ImageDataGenerator

from art.data_generators import KerasDataGenerator
from art.defences.trainer import AdversarialTrainerFBFPyTorch
from art.utils import load_cifar10

import torch.nn.functional as F
from art.classifiers import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn
import math

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
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
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
    return PreActResNet(PreActBlock, [2,2,2,2])

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

cifar_mu = np.ones((3, 32, 32))
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465

cifar_std = np.ones((3, 32, 32))
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616

x_train = x_train.transpose(0, 3, 1, 2).astype('float32')
x_test = x_test.transpose(0, 3, 1, 2).astype('float32')

np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)
model = PreActResNet18()

# model = PreActResNet18().cuda()
model.apply(initialize_weights)
model.train()

opt = torch.optim.SGD(model.parameters(), lr=0.21, momentum=0.9, weight_decay=5e-4)

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

epsilon = (8.0 / 255.)

trainer = AdversarialTrainerFBFPyTorch(classifier, eps=epsilon, use_amp=False)

# Build a Keras image augmentation object and wrap it in ART
batch_size = 128
datagen = ImageDataGenerator(
    horizontal_flip=True
)

datagen.fit(x_train)
art_datagen = KerasDataGenerator(
    datagen.flow(x=x_train, y=y_train, batch_size=batch_size, shuffle=True),
    size=x_train.shape[0],
    batch_size=batch_size,
)


trainer.fit_generator(art_datagen, nb_epochs=30)

