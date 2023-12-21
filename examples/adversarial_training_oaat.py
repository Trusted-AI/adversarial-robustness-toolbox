"""
This is an example of how to use ART for adversarial training of a model with OAAT protocol
for defence against larger perturbations.
"""

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from art.estimators.classification import PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.defences.trainer import AdversarialTrainerOAATPyTorch
from art.utils import load_cifar10
from art.attacks.evasion import ProjectedGradientDescent

"""
For this example we choose the PreActResNet18 model as used in the paper
(Paper link: https://link.springer.com/chapter/10.1007/978-3-031-20065-6_18)
The code for the model architecture has been adopted from
https://github.com/val-iisc/OAAT/blob/main/models/resnet.py
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


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

x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
)

dataset = CIFAR10_dataset(x_train, y_train, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

train_params = {
    "weight_decay": 5e-4,
    "lr": 0.2,
    "momentum": 0.9,
    "norm": np.inf,
    "i_epsilon": 4 / 255.0,
    "epsilon": 16 / 255.0,
    "alternate_iter_eps": 12 / 255.0,
    "mixup_epsilon": 24 / 255.0,
    "max_iter": 10,
    "alpha": 1.0,
    "mixup_alpha": 0.2,
    "beta": 1.5,
    "beta_final": 3.0,
    "awp_gamma": 0.005,
    "awp_warmup": 10,
    "lr_schedule": "cosine",
    "lpips_weight": 1.0,
    "i_lpips_weight": 1.0,
    "oaat_warmup": 0,
    "swa_save_epoch": 0,
    "list_swa_tau": [0.995],
    "list_swa_epoch": [0],
    "models_path": None,
    "load_swa_model_tau": 0.995,
    "layer_names_activation": ["layer1", "layer2", "layer3", "layer4"],
}


# Step 2: create the PyTorch model
model = ResNet18()
opt = torch.optim.SGD(
    model.parameters(),
    lr=train_params["lr"],
    momentum=train_params["momentum"],
    weight_decay=train_params["weight_decay"],
)

proxy_model = ResNet18()
proxy_opt = torch.optim.SGD(proxy_model.parameters(), lr=train_params["awp_gamma"])

lpips_model = ResNet18()

criterion = nn.CrossEntropyLoss()

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=opt,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

proxy_classifier = PyTorchClassifier(
    model=proxy_model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=proxy_opt,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

lpips_classifier = PyTorchClassifier(
    model=lpips_model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    input_shape=(3, 32, 32),
    nb_classes=10,
)


list_avg_models = [
    PyTorchClassifier(
        model=ResNet18(),
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    for _ in range(len(train_params["list_swa_tau"]))
]


attack = ProjectedGradientDescent(
    classifier,
    norm=train_params["norm"],
    eps=train_params["epsilon"],
    eps_step=train_params["epsilon"] / 4.0,
    max_iter=train_params["max_iter"],
    targeted=False,
    num_random_init=1,
    batch_size=128,
    verbose=False,
)

# Step 4: Create the trainer object - AdversarialTrainerOAATPyTorch
trainer = AdversarialTrainerOAATPyTorch(
    classifier, proxy_classifier, lpips_classifier, list_avg_models, attack, train_params=train_params
)


# Build a PyTorch data generator in ART
art_datagen = PyTorchDataGenerator(iterator=dataloader, size=x_train.shape[0], batch_size=128)

# Step 5: fit the trainer
trainer.fit_generator(art_datagen, validation_data=(x_test, y_test), nb_epochs=110)

x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
print(
    "Accuracy on benign test samples after adversarial training: %.2f%%"
    % (np.sum(x_test_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
)

attack_test_eps8s2 = ProjectedGradientDescent(
    classifier,
    norm=np.inf,
    eps=8.0 / 255.0,
    eps_step=2.0 / 255.0,
    max_iter=20,
    targeted=False,
    num_random_init=1,
    batch_size=128,
    verbose=False,
)
x_test_attack_eps8s2 = attack_test_eps8s2.generate(x_test, y=y_test)
x_test_attack_pred_eps8s2 = np.argmax(classifier.predict(x_test_attack_eps8s2), axis=1)
print(
    "Accuracy on original PGD adversarial samples with epsilon 8/255 and step size 2/255 after "
    "adversarial training: %.2f%%"
    % (np.sum(x_test_attack_pred_eps8s2 == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
)

attack_test_eps16s4 = ProjectedGradientDescent(
    classifier,
    norm=np.inf,
    eps=16.0 / 255.0,
    eps_step=4.0 / 255.0,
    max_iter=20,
    targeted=False,
    num_random_init=1,
    batch_size=128,
    verbose=False,
)
x_test_attack_eps16s4 = attack_test_eps16s4.generate(x_test, y=y_test)
x_test_attack_pred_eps16s4 = np.argmax(classifier.predict(x_test_attack_eps16s4), axis=1)
print(
    "Accuracy on original PGD adversarial samples with epsilon 16/255 and step size 4/255 after "
    "adversarial training: %.2f%%"
    % (np.sum(x_test_attack_pred_eps16s4 == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
)
