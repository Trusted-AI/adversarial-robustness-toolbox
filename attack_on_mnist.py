import keras

import tensorflow as tf

from src.classifiers import cnn
from src.utils import load_mnist

BATCH_SIZE = 128
NB_EPOCHS = 10

session = tf.Session()
keras.backend.set_session(session)

# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()

im_shape = X_train[0].shape

# learn with bounded relu
model = cnn.cnn_model(im_shape,act="brelu")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train,Y_train,epochs=NB_EPOCHS,batch_size=BATCH_SIZE)

scores = model.evaluate(X_test,Y_test)

print("\naccuracy: %.2f%%" % (scores[1] * 100))