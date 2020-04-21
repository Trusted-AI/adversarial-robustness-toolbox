from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from art.defences.trainer import AdversarialTrainerFBF
from art.attacks.evasion import ProjectedGradientDescent
import time

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')

input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

start_start_time = time.time()
classifier = KerasClassifier(clip_values=(min_, max_), model=model, use_logits=False)
trainer = AdversarialTrainerFBF(classifier, eps=0.3)
trainer.fit(x_train, y_train)
train_time = time.time()
print('Time taken(sec): ',train_time-start_start_time)

x_test_robust_pred = np.argmax(classifier.predict(x_test), axis=1)
nb_correct_robust_pred = np.sum(x_test_robust_pred == np.argmax(y_test, axis=1))

classifier.save('mnist-robust.h5','./')
print("Original test data (first 100 images):")
print("accuracy: {}".format(nb_correct_robust_pred/x_test.shape[0]))

acc = []
for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    eps_step = (1.5 * eps) / 40
    attack_test = ProjectedGradientDescent(classifier=classifier, norm=np.inf, eps=eps,
                                           eps_step=eps_step, max_iter=40, targeted=False,
                                           num_random_init=5, batch_size=32)
    x_test_attack = attack_test.generate(x_test)
    x_test_attack_pred = np.argmax(classifier.predict(x_test_attack), axis=1)
    nb_correct_attack_pred = np.sum(x_test_attack_pred == np.argmax(y_test, axis=1))
    acc.append(nb_correct_attack_pred / x_test.shape[0])

print(acc)
np.save('acc.npy',acc)