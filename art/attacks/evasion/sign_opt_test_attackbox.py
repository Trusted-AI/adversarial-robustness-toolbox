"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from matplotlib import pyplot as plt

# from art.attacks.evasion import FastGradientMethod, BoundaryAttack
from sign_opt import SignOPTAttack
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from tests.attacks.utils import random_targets

# Step 0: Define the neural network model, return logits instead of activation in forward method
###
# define the network as https://arxiv.org/pdf/1608.04644.pdf,
###
class Net_table1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # https://discuss.pytorch.org/t/calculation-for-the-input-to-the-fully-connected-layer/82774/11
        # x.shape: torch.Size([128, 64, 4, 4]) so, 64*4*4
        self.fc_1 = nn.Linear(in_features= 1024, out_features=200) 
        self.fc_2 = nn.Linear(in_features=200, out_features=200)
        self.fc_3 = nn.Linear(in_features=200, out_features=10)
        
    # https://github.com/Carco-git/CW_Attack_on_MNIST/blob/master/MNIST_Model.py
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 1024) 
        
        x = F.relu(self.fc_1(x))
        x = F.dropout(x, p=0.5) 
        x = F.relu(self.fc_2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc_3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024,200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200,200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.Conv2d(32, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)
# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
# Step 2: Create the model
model = Net()
# critical to use DataParallel() b/c the model is trained with dataparallel
model = torch.nn.DataParallel(model, device_ids=[0])
# Step 2a: Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Specify a path for loading model to
PATH = "mnist_gpu.pt"

# Load model
device = torch.device('cpu')


model.load_state_dict(torch.load(PATH, map_location=device), strict=False)
# print(model)
model.train(mode=False)
# pred = model(torch.from_numpy(x_test[0:5]))

      
# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type = "cpu",
)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples

# read variable from parameter
import sys

e = 1.5
q = 14000
targeted = False
length = 100
clipping = True
if len(sys.argv) == 6:
    e = float(sys.argv[1])
    q = int(sys.argv[2])
    targeted = eval(sys.argv[3])
    start_index = int(sys.argv[4])
    clipping = eval(sys.argv[5])
    
if targeted:
    attack = SignOPTAttack(estimator=classifier, 
                           targeted=targeted,
                           max_iter=5000, query_limit=40000, 
                           eval_perform=True, verbose=False, 
                           clipped=clipping)
else:
    attack = SignOPTAttack(estimator=classifier, 
                           targeted=targeted, 
                           query_limit=q, 
                           eval_perform=True, verbose=False, 
                           clipped=clipping)
targets = random_targets(y_test, attack.estimator.nb_classes)

if targeted:
    x_test_adv = attack.generate(x=x_test, y=targets)
else:   
    x_test_adv = attack.generate(x=x_test[:length])


# calculate performace
# For untargeted attack, we only consider examples that are correctly predicted by model
model_failed = 0
L2 = attack.logs.sum()/length
count = 0
for l2 in attack.logs:
    if l2 <=e and l2 != 0.0:
        count += 1
SR = ((count - model_failed)/length)*100
# print(f'Avg l2 = {L2}, Success Rate={SR}% with e={e} and {length} examples')


# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:length], axis=1)) / len(y_test[:length])
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
print(f'{q}, {e}, {round(L2,2)}, {SR}%, {clipping}, {accuracy*100}%')


