"""
The script demonstrates a simple example of using ART with MXNet. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import mxnet
from mxnet.gluon.nn import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import MXClassifier
from art.utils import load_mnist

# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to MXNet's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2))
x_test = np.transpose(x_test, (0, 3, 1, 2))

# Step 2: Create the model

model = mxnet.gluon.nn.Sequential()
with model.name_scope():
    model.add(Conv2D(channels=4, kernel_size=5, activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=1))
    model.add(Conv2D(channels=10, kernel_size=5, activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=1))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10))
    model.initialize()

loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
trainer = mxnet.gluon.Trainer(model.collect_params(), "adam", {"learning_rate": 0.01})

# Step 3: Create the ART classifier

classifier = MXClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=loss,
    input_shape=(28, 28, 1),
    nb_classes=10,
    optimizer=trainer,
    ctx=None,
    channels_first=True,
    preprocessing_defences=None,
    preprocessing=(0.0, 1.0),
)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
