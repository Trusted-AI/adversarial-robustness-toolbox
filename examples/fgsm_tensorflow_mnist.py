"""
This tutorial uses the stable release of TensorFlow v2.0 dated 4 October 2019. Here, a simple classifier 
is built using the functional API as TensorFlow v2.0 is now tightly-knit with Keras and provides well-rounded 
support. Adversarial examples are generated using the gradients of the model. 

Note: The hyper-parameters are not fully optimised for accuracy. Eager executing is disabled so as to use the
KerasClassifier instance

Author: Rishabh Anand (GitHub: @rish-16)
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.utils import to_categorical

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_mnist

# disabling eager execution
print ('TensorFlow version: {}'.format(tf.__version__))
tf.compat.v1.disable_v2_behavior()

# preprocessing MNIST
(x_train, y_train), (x_test, y_test), min_pixel, max_pixel = load_mnist()

# building a ConvNet
conv_input = Input(shape=[28, 28, 1])
x = Conv2D(32, (3,3), activation="relu")(conv_input)
x = MaxPooling2D(pool_size=(2,2))(conv_input)
x = Conv2D(64, (3,3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
conv_out = Dense(10, activation="softmax")(x)

# compiling the model. Tracking the "accuracy" metric
model = Model(inputs=conv_input, outputs=conv_out)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])

# visualising model architecture
model.summary()

# building the classifier using ART
mnist_net = KerasClassifier(model, clip_values=(min_pixel, max_pixel))

# training the classifier
training_history = mnist_net.fit(x_train, y_train, batch_size=128, nb_epochs=10)

# getting accuracy of classifier
def evaluate(predictions, labels):
	accuracy = 0
	for i in range(len(predictions)):
		if np.argmax(predictions[i]) == np.argmax(labels[i]):
			accuracy += 1

	return float(accuracy / len(predictions)) * 100

# testing the classifier
predictions = mnist_net.predict(x_test)
acc_before = evaluate(predictions, y_test)
print ('Accuracy before attack: {}'.format(acc_before))

# creating adversarial examples using the classifier's gradients
epsilon = 0.2
adversarial_crafter = FastGradientMethod(mnist_net, eps=epsilon)
adv_x_test = adversarial_crafter.generate(x=x_test)

# testing model on adversarial examples
print ('Implementing attack on KerasClassifier model...')
adv_preds = mnist_net.predict(adv_x_test)
acc_after = evaluate(adv_preds, y_test)
print ('Accuracy after attack: {}'.format(acc_after))