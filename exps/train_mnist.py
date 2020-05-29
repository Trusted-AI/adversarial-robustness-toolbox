from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import h5py
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from art.attacks.evasion import ProjectedGradientDescent
import numpy as np

import time

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')

input_shape = (28, 28, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=30,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save('mnist-keras-2-1-5.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

classifier = KerasClassifier(clip_values=(min_, max_), model=model, use_logits=False)

acc = [score[1]]
for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    eps_step = (1.5 * eps) / 40
    attack_test = ProjectedGradientDescent(classifier=classifier, norm=np.inf, eps=eps,
                                           eps_step=eps_step, max_iter=40, targeted=False,
                                           num_random_init=5, batch_size=32)
    x_test_attack = attack_test.generate(x_test)
    x_test_attack_pred = np.argmax(classifier.predict(x_test_attack), axis=1)
    nb_correct_attack_pred = np.sum(x_test_attack_pred == np.argmax(y_test, axis=1))
    acc.append(nb_correct_attack_pred / x_test.shape[0])
    print(acc, flush=True)

print(acc)
np.save('./exps/acc_orig.npy',acc)

