"""Train a PyTorch classifier on MNIST dataset, then attack it with targeted universal adversarial perturbations."""

from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from art.attacks.evasion import TargetedUniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format
x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model
model = nn.Sequential(
    nn.Conv2d(1, 4, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(4, 10, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(4 * 4 * 10, 100),
    nn.Linear(100, 10),
)

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
)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)

# Step 5: Create a one-hot encoded target label array, specifying a specific class as the target for the attack.
TARGET = 0
y_target = np.zeros([len(x_train), 10])
for i in range(len(x_train)):
    y_target[i, TARGET] = 1.0

# Step 6: Run Targeted Universal Perturbation attack
attack = TargetedUniversalPerturbation(
    classifier,
    max_iter=1,
    attacker="fgsm",
    attacker_params={"delta": 0.4, "eps": 0.01, "targeted": True, "verbose": False},
)
x_train_adv = attack.generate(x_train, y=y_target)

# Step 7: Print attack statistics
print("Attack statistics:")
print(f"Fooling rate: {attack.fooling_rate:.2%}")
print(f"Targeted success rate: {attack.targeted_success_rate:.2%}")
print(f"Converged: {attack.converged}")

# Step 7a: Evaluate the attack results
train_y_pred = np.argmax(classifier.predict(x_train_adv), axis=1)
print("\nMisclassified train samples:", np.sum(np.argmax(y_train, axis=1) != train_y_pred))

# Step 8: Generate adversarial examples for test set
x_test_adv = x_test + attack.noise

# Step 8a: Evaluate the attack results on the test set
test_y_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
print("Misclassified test samples:", len(x_test_adv[np.argmax(y_test, axis=1) != test_y_pred]))

# Step 9: Plot some misclassified samples
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(x_test_adv[i, ...].squeeze())
    ax.axis("off")
    ax.text(
        0.5,
        -0.05,
        f"True Label: {np.argmax(y_test, axis=1)[i]}, Predicted Label: {test_y_pred[i]}",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
    )

plt.tight_layout()
plt.suptitle("Adversarial example and labels", fontsize=16, y=1.01)
plt.show()
