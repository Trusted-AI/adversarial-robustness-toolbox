from __future__ import print_function
import sys
sys.path.append('/Users/ambrish/github/adversarial-robustness-toolbox/')
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from art.defences.trainer import AdversarialTrainerFB


model = load_model('mnist-keras-2-1-5.h5')
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


classifier = KerasClassifier(clip_values=(min_, max_), model=model, use_logits=False)
trainer = AdversarialTrainerFB(classifier, nb_epochs=1, batch_size=50, ratio=0.5)
trainer.fit(x_train, y_train)

x_test_robust_pred = np.argmax(classifier.predict(x_test), axis=1)
nb_correct_robust_pred = np.sum(x_test_robust_pred == np.argmax(y_test, axis=1))

classifier.save('mnist-robust.h5','./')
print("Original test data (first 100 images):")
print("accuracy: {}".format(nb_correct_robust_pred/x_test.shape[0]))
