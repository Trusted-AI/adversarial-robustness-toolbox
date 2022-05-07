"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
from tabnanny import verbose
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from matplotlib import pyplot as plt

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


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type="cpu",
)

# Step 4: Train the ART classifier; If model file exist, load model from file
ML_model_Filename = "Pytorch_Model_art.pkl"

# Load the Model back from file
try:
    with open(ML_model_Filename, "rb") as file:
        classifier = pickle.load(file)
except FileNotFoundError:
    print("No existing model, training the model instead")
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
    # Save the model
    with open(ML_model_Filename, "wb") as file:
        pickle.dump(open, file)


# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples

import sys

e = 1.5
q = 4000
targeted = True
start_index = 0
length = 100
clipping = True
if len(sys.argv) == 6:
    e = float(sys.argv[1])
    q = int(sys.argv[2])
    targeted = eval(sys.argv[3])
    start_index = int(sys.argv[4])
    clipping = eval(sys.argv[5])
print(
    f"parameters: e={e}, query limitation={q}, targeted attack={targeted}, length of examples={length}, start_index={start_index}, clipped={clipping}"
)

if targeted:
    attack = SignOPTAttack(
        estimator=classifier,
        targeted=targeted,
        max_iter=5000,
        query_limit=40000,
        eval_perform=True,
        verbose=False,
        clipped=clipping,
    )
else:
    attack = SignOPTAttack(
        estimator=classifier, targeted=targeted, query_limit=q, eval_perform=True, verbose=False, clipped=clipping
    )


targets = random_targets(y_test, attack.estimator.nb_classes)
end_index = start_index + length
x = x_test[start_index:end_index]
targets = targets[start_index:end_index]
x_test_adv = attack.generate(x=x, y=targets, x_init=x_train)


def plot_image(x):
    for i in range(len(x[:])):
        pixels = x[i].reshape((28, 28))
        plt.imshow(pixels, cmap="gray")
        plt.show()


# plot_image(x_test_adv)

# calculate performace
# For untargeted attack, we only consider examples that are correctly predicted by model
model_failed = 0
L2 = attack.logs.sum() / length

count = 0
for l2 in attack.logs:
    if l2 <= e and l2 != 0.0:
        count += 1
SR = ((count - model_failed) / length) * 100

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:length], axis=1)) / len(y_test[:length])

print(f"{q}, {e}, {round(L2,2)}, {round(SR,2)}%, {clipping}, {accuracy*100}%")
