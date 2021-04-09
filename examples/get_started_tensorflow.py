"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import tensorflow.compat.v1 as tf
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowClassifier
from art.utils import load_mnist

# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 2: Create the model

input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
labels_ph = tf.placeholder(tf.int32, shape=[None, 10])

x = tf.layers.conv2d(input_ph, filters=4, kernel_size=5, activation=tf.nn.relu)
x = tf.layers.max_pooling2d(x, 2, 2)
x = tf.layers.conv2d(x, filters=10, kernel_size=5, activation=tf.nn.relu)
x = tf.layers.max_pooling2d(x, 2, 2)
x = tf.layers.flatten(x)
x = tf.layers.dense(x, 100, activation=tf.nn.relu)
logits = tf.layers.dense(x, 10)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Step 3: Create the ART classifier

classifier = TensorFlowClassifier(
    clip_values=(min_pixel_value, max_pixel_value),
    input_ph=input_ph,
    output=logits,
    labels_ph=labels_ph,
    train=train,
    loss=loss,
    learning=None,
    sess=sess,
    preprocessing_defences=[],
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
