"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from matplotlib import pyplot as plt

from art.attacks.evasion import FastGradientMethod, BoundaryAttack
from sign_opt import SignOPTAttack
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from tests.attacks.utils import random_targets

# Step 0: Define the neural network model, return logits instead of activation in forward method

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 4 * 4 * 10)
        
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

###
# define the network as https://arxiv.org/pdf/1608.04644.pdf, table 1
###
class Net_table1(nn.Module):
    def __init__(self):
        super(Net_table1, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # https://discuss.pytorch.org/t/calculation-for-the-input-to-the-fully-connected-layer/82774/11
        # x.shape: torch.Size([128, 64, 4, 4]) so, 64*4*4
        self.fc_1 = nn.Linear(in_features= 64*4*4 , out_features=200) 
        self.fc_2 = nn.Linear(in_features=200, out_features=10)
        
    # https://github.com/Carco-git/CW_Attack_on_MNIST/blob/master/MNIST_Model.py
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 64*4*4 ) 
        
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
# (60000, 28, 28, 1) (60000, 10) (10000, 28, 28, 1) (10000, 10) 0.0 1.0
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, min_pixel_value, max_pixel_value)

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
# (60000, 1, 28, 28) (10000, 1, 28, 28)
# print(x_train.shape, x_test.shape)

# Step 2: Create the model

model_table1 = Net_table1()
model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer_table1 = optim.Adam(model_table1.parameters(), lr=0.01)

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

classifier_table1 = PyTorchClassifier(
    model=model_table1,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=model_table1,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type = "cpu",
)

# Step 4: Train the ART classifier; If model file exist, load model from file

classifier_table1.fit(x_train, y_train, batch_size=64, nb_epochs=3)

ML_model_Filename = "Pytorch_Model.pkl"  
# Load the Model back from file
try:
    with open(ML_model_Filename, 'rb') as file:  
        classifier = pickle.load(file)
except FileNotFoundError:
    print('No existing model, training the model instead')
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
    # Save the model
    with open(ML_model_Filename, 'wb') as file:  
        pickle.dump(classifier, file)

# print(classifier)
# exit()
# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
# print(f'{predictions.shape}')

# Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
# attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=0, delta=0.001, epsilon=0.001)

# read variable from parameter
import sys
# Display File name 
print("Script name ", sys.argv[0])

e = 1.5
q = 4000
target = False
start_index = 0
if len(sys.argv) == 7:
    print(f'e={sys.argv[1]}, q={sys.argv[2]}, targeted={sys.argv[3]}, length={sys.argv[4]}, eps={sys.argv[5]}, start_inde={sys.argv[6]}')
    e = float(sys.argv[1])
    q = int(sys.argv[2])
    target = eval(sys.argv[3])
    start_index = int(sys.argv[4])
else:
    print("parameters: e(=1.5), query limitation(=4000), targeted attack(=False), length of examples(=100), start_index(=0)")


test_targeted = target
if test_targeted:
    attack = SignOPTAttack(estimator=classifier, targeted=True, max_iter=5000, query_limit=40000, eval_perform=True)
else:
    attack = SignOPTAttack(estimator=classifier, targeted=test_targeted, query_limit=q, eval_perform=True)
length = 3 #len(x_test) #
print(f'test targeted = {test_targeted}, length={length}')
targets = random_targets(y_test, attack.estimator.nb_classes)
end_index = start_index+length
x = x_test[start_index: end_index]
targets = targets[start_index: end_index]
x_test_adv = attack.generate(x=x, targets=targets, x_train=x_train)

def plot_image(x):
    for i in range(len(x[:])):
        # print(i)
        pixels = x[i].reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()
# plot_image(x_test_adv)

# calculate performace
# For untargeted attack, we only consider examples that are correctly predicted by model
model_failed = 0
for i in range(len(x)):
    if attack._is_label(x_test[i+start_index], np.argmax(y_test[i+start_index])) == False:
        model_failed += 1
        attack.logs[i] = 0
        attack.logs_torch[i] = 0
        print(f'index={i+start_index}, y_test={np.argmax(y_test[i+start_index])}, predict label={attack._predict_label(x_test[i+start_index])}')


if model_failed > 0:
    length -= model_failed
    print(f'length is adjusted with {model_failed} failed prediction')
    
L2 = attack.logs.sum()/length

SR = ((attack.logs <= e).sum() - model_failed)/length
print(f'Avg l2 = {L2}, Success Rate={SR} with e={e} and {length} examples')

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:length], axis=1)) / len(y_test[:length])
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
