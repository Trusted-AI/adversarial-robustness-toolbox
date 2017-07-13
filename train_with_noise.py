from config import DATA_PATH, config_dict

import os
import numpy as np

import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

from src.classifiers.cnn import CNN
from src.classifiers.resnet import ResNet
from src.classifiers.utils import save_classifier, load_classifier
from src.utils import get_args, get_verbose_print, load_dataset, make_directory, set_group_permissions_rec

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)

v_print = get_verbose_print(args.verbose)

comp_params = {"loss": 'categorical_crossentropy',
               "optimizer": 'adam',
               "metrics": ['accuracy']}

# --------------------------------------------------------------------------------------------- GET CLASSIFIER

# get dataset
(X_train, Y_train), (X_test, Y_test) = load_dataset(args.dataset)
# X_train, Y_train, X_test, Y_test = X_train[:1000], Y_train[:1000], X_test[:1000], Y_test[:1000]

im_shape = X_train[0].shape

session = tf.Session()
K.set_session(session)

if args.save is not False:

    if args.save:
        MODEL_PATH = os.path.abspath(args.save)

    else:
        MODEL_PATH = os.path.join(os.path.abspath(DATA_PATH), "classifiers", args.dataset, args.classifier, args.act,
                                  "gaussian", "stdev%.2f" % args.std_dev, "pert-insts%d" % args.nb_instances)

    v_print("Classifier saved in", MODEL_PATH)

if args.classifier == "cnn":
    classifier = CNN(im_shape, act=args.act, bnorm=False)

elif args.classifier == "resnet":
    classifier = ResNet(im_shape, act=args.act, bnorm=False)

# Fit the classifier

classifier.compile(comp_params)

if args.save is not False:

    make_directory(MODEL_PATH)

    checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, "best-weights.h5"), monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # Remote monitor
    monitor = TensorBoard(log_dir=os.path.join(MODEL_PATH, 'logs'), write_graph=False)

    callbacks_list = [checkpoint, monitor]
else:
    callbacks_list = []

# generate gaussian perturbed instances
x_gau_perts = np.random.normal(X_train, scale=args.std_dev, size=(args.nb_instances, )+X_train.shape)
x_gau_perts = x_gau_perts.reshape(-1, *im_shape)
y_gau_perts = np.tile(Y_train, (args.nb_instances, 1))

classifier.fit(np.vstack((X_train, x_gau_perts)), np.vstack((Y_train, y_gau_perts)), verbose=2*int(args.verbose),
          validation_split=args.val_split, epochs=args.nb_epochs, batch_size=args.batch_size, callbacks=callbacks_list)

if args.save is not False:
    save_classifier(classifier, MODEL_PATH)
    # Load classifier with best validation score
    classifier = load_classifier(MODEL_PATH, "best-weights.h5")

    # Change files' group and permissions if on ccc
    if config_dict['profile'] == "CLUSTER":
        set_group_permissions_rec(MODEL_PATH)

scores = classifier.evaluate(X_test, Y_test, verbose=args.verbose)
v_print("\naccuracy: %.2f%%" % (scores[1] * 100))
