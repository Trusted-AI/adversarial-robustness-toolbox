# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function

import os

from config import DATA_PATH, config_dict
import keras.backend as k
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import numpy as np
import tensorflow as tf

from art.classifiers.cnn import CNN
from art.classifiers.mlp import MLP
from art.classifiers.resnet import ResNet
from art.classifiers.utils import save_classifier, load_classifier
from art.utils import get_args, get_verbose_print, load_dataset, make_directory

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__, options="bcdefrsvz")

v_print = get_verbose_print(args.verbose)

comp_params = {"loss": 'categorical_crossentropy',
               "optimizer": 'adam',
               "metrics": ['accuracy']}

# --------------------------------------------------------------------------------------------- GET CLASSIFIER

# Get dataset
(X_train, Y_train), (X_test, Y_test), _, _ = load_dataset(args.dataset)

if os.path.isfile(args.dataset):
    X_train = np.load(args.dataset)
    Y_train = Y_train if "train.npy" in args.dataset else Y_test
im_shape = X_train[0].shape

session = tf.Session()
k.set_session(session)

if args.classifier == "cnn":
    classifier = CNN(im_shape, act=args.act, bnorm=False, defences=args.defences, dataset=args.dataset)
elif args.classifier == "resnet":
    classifier = ResNet(im_shape, act=args.act, bnorm=False, defences=args.defences)
elif args.classifier == "mlp":
    classifier = MLP(im_shape, act=args.act, bnorm=False, defences=args.defences, dataset=args.dataset)

# Fit the classifier
classifier.compile(comp_params)

if args.save is not False:
    if args.save:
        MODEL_PATH = os.path.abspath(args.save)
    else:
        if args.defences:
            defences = "-".join(args.defences)
        else:
            defences = ""
        MODEL_PATH = os.path.join(os.path.abspath(DATA_PATH), "classifiers", args.dataset, args.classifier, args.act,
                                  defences)

    v_print("Classifier saved in", MODEL_PATH)
    make_directory(MODEL_PATH)

    # Save best classifier weights
    # checkpoint = ModelCheckpoint(os.path.join(FILEPATH,"best-weights.{epoch:02d}-{val_acc:.2f}.h5"),
    #                              monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, "best-weights.h5"), monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # Remote monitor
    monitor = TensorBoard(log_dir=os.path.join(MODEL_PATH, 'logs'), write_graph=False)
    callbacks_list = [checkpoint, monitor]
else:
    callbacks_list = []

callbacks_list.append(EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='max'))

classifier.fit(X_train, Y_train, verbose=2*int(args.verbose), validation_split=args.val_split, epochs=args.nb_epochs,
               batch_size=args.batch_size, callbacks=callbacks_list)

if args.save is not False:
    save_classifier(classifier, MODEL_PATH)

    # Load model with best validation score
    classifier = load_classifier(MODEL_PATH, "best-weights.h5")

scores = classifier.evaluate(X_test, Y_test, verbose=args.verbose)
v_print("\naccuracy: %.2f%%" % (scores[1] * 100))
