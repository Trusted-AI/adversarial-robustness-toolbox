#### CELL 1
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from art.utils import compute_accuracy
from art.estimators.classification import PyTorchClassifier
from art.estimators.certification.randomized_smoothing import PyTorchRandomizedSmoothing
from art.data_generators import PyTorchDataGenerator

import numpy as np
import matplotlib.pyplot as plt

#### CELL 2
print(torch.__version__)

#### CELL 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#### CELL 4
batch_size = 64
train_data = datasets.CIFAR10(
    "./dataset_cache", 
    train=True, 
    download=True, 
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
)
test_data = datasets.CIFAR10(
    "./dataset_cache", 
    train=False, 
    download=True, 
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_data, 
    shuffle=True, 
    batch_size=batch_size, 
    num_workers=1
)
test_loader = DataLoader(
    test_data, 
    shuffle=False, 
    batch_size=batch_size,
    num_workers=1, 
    pin_memory=True
)

#### CELL 6
num_train_samples = 50000

x_train = torch.zeros((num_train_samples, 3, 32, 32), dtype=torch.float32)
y_train = torch.zeros((num_train_samples,), dtype=torch.uint8)

for i,(data,labels) in enumerate(train_loader):
    x_train[(i) * batch_size : (i+1) * batch_size, :, :, :] = data
    y_train[(i) * batch_size : (i+1) * batch_size] = labels

#### CELL 7
num_train_samples = 10000

x_test = torch.zeros((num_train_samples, 3, 32, 32), dtype=torch.float32)
y_test = torch.zeros((num_train_samples,), dtype=torch.uint8)

for i,(data,labels) in enumerate(test_loader):
    x_test[(i) * batch_size : (i+1) * batch_size, :, :, :] = data
    y_test[(i) * batch_size : (i+1) * batch_size] = labels


#### CELL 8
def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, width=1, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width, layers[0])
        self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion * width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet110(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], width=1, **kwargs)
    return model

#### CELL 9
def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, width=1, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width, layers[0])
        self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion * width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet110(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], width=1, **kwargs)
    return model


#### CELL 10
model = resnet110()

#### CELL 11
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
loss = nn.CrossEntropyLoss()

#### CELL 12
# SmoothMix params
sigma = 0.25
alpha = 0.5
mix_step = 1
eta = 5.0
num_noise_vec = 2
warmup = 10
num_steps = 8
mix_step = 0 # 0 for normal, 1 for smoothmix with one-step adv
maxnorm_s = maxnorm = None

rs_smoothmix_classifier = PyTorchRandomizedSmoothing(
    model=model,
    loss=loss,
    input_shape=(3, 32, 32),
    nb_classes=10,
    optimizer=optimizer,
    scheduler=scheduler,
    clip_values=(0.0, 255.0),
    device_type=device,
    alpha=alpha,
    scale=sigma, 
    num_noise_vec=num_noise_vec,
    train_multi_noise=True,
    attack_type="PGD",
    num_steps=num_steps,
    warmup=warmup,
    eta=eta,
    mix_step=mix_step,
    maxnorm_s=maxnorm_s,
    maxnorm=maxnorm
)

#### CELL 13
print("Start training...")
rs_smoothmix_classifier.fit(
    x_train, 
    y_train, 
    nb_epochs=150, 
    batch_size=256, 
    train_method='smoothmix'
)

#### CELL 14
print("Predictions for Trained Model")
y_test_encoded = F.one_hot(y_test.to(torch.int64))
x_preds_rs_1 = rs_smoothmix_classifier.predict(x_test[:500])
acc_rs_1, cov_rs_1 = compute_accuracy(x_preds_rs_1, y_test_encoded[:500].numpy())
print("\nSmoothMix Classifier, sigma=" + str(sigma))
print("Accuracy: {}".format(acc_rs_1))
print("Coverage: {}".format(cov_rs_1))

#### CELL 15
print("Certified Radius for Single Image")
def getCertAcc(radius, pred, y_test):
    """
    Calculate certification accuracy for a given radius
    """
    rad_list = np.linspace(0, 2.25, 201)
    cert_acc = []
    num_cert = len(radius)
    for r in rad_list:
        rad_idx = np.where(radius >= r)[0]
        y_test_subset = y_test[rad_idx]
        cert_acc.append(np.sum(pred[rad_idx] == y_test_subset) / num_cert)
    return cert_acc

def calculateACR(target, prediction, radius):
    tot = 0
    cnt = 0
    for i in range(0, len(prediction)):
        if(prediction[i] == target[i]):
            tot += radius[i]
        cnt += 1
    return tot/cnt

print("Single image certification return certified radius, index or random) ")
import random 
index = random.randint(0,9999)
x_sample = x_test[index].expand((1,3,32,32))
prediction, radius = rs_smoothmix_classifier.certify(x_sample, n = 100000)
print("Prediction: {} and Radius: {}".format(prediction,radius))

#### CELL 16
print("Certification on Test Images")
# no.of test images for ACR/graph (ACR inside the graph)
start_img = 500
num_img = 500
skip = 1
N = 100000

prediction_1, radius_1 = rs_smoothmix_classifier.certify(
    x_test[(start_img-1):(start_img-1)+(num_img*skip):skip], 
    n=N
)
acr = calculateACR(
    target=np.array(y_test[(start_img-1):(start_img-1)+(num_img*skip):skip]), 
    prediction=np.array(prediction_1), 
    radius=np.array(radius_1)
)
print("ACR for Smooth Adversarial Classifier: ", acr)

#### CELL 17
print("Plotting...")
rad_list = np.linspace(0, 2.25, 201)
plt.plot(rad_list, getCertAcc(radius_1, prediction_1, np.array(y_test)), 'r-', label='smoothed, $\sigma=$' + str(sigma))
plt.xlabel('L2 radius')
plt.ylabel('Certified Accuracy')
plt.legend()
plt.title('Average Certified Radius plot: ACR {}'.format(acr))
# plt.show()
plt.savefig("./results.png")