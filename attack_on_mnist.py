import keras

import tensorflow as tf

from cleverhans.attacks import fgsm
from cleverhans.utils_tf import model_train, model_eval, batch_eval

from src.classifiers import cnn
from src.utils import load_mnist

BATCH_SIZE = 128
NB_EPOCHS = 10

#MODEL TRAINING
# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()
# X_train, Y_train, X_test, Y_test = X_train[:1000], Y_train[:1000], X_test[:1000], Y_test[:1000]

im_shape = X_train[0].shape

session = tf.Session()
keras.backend.set_session(session)

# learn with bounded relu
model = cnn.cnn_model(im_shape,act="brelu")

# Fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=NB_EPOCHS,batch_size=BATCH_SIZE)

scores = model.evaluate(X_test,Y_test)

print("\naccuracy: %.2f%%" % (scores[1] * 100))

# ATTACK
#Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None,im_shape[0],im_shape[1],im_shape[2]))
y = tf.placeholder(tf.float32, shape=(None,10))

predictions = model(x)

# craft adversarials with Fast Gradient Sign Method (FGSM)
adv_x = fgsm(x, predictions, eps=0.1)

eval_params = {'batch_size': BATCH_SIZE}
X_test_adv = batch_eval(session, [x], [adv_x], [X_test], args=eval_params)

scores = model.evaluate(X_test_adv,Y_test)

print("\naccuracy on adversarials: %.2f%%" % (scores[1] * 100))