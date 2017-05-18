from config import DATA_PATH

import json
import numpy as np
import os

import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

import tensorflow as tf

from cleverhans.attacks import FastGradientMethod

from src.classifiers import cnn
from src.utils import get_args, get_verbose_print, load_mnist, make_directory

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)

v_print = get_verbose_print(args.verbose)

comp_params = {"loss":'categorical_crossentropy',
               "optimizer":'adam',
               "metrics":['accuracy']}

# --------------------------------------------------------------------------------------------- GET CLASSIFIER
# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()
# X_train, Y_train, X_test, Y_test = X_train[:1000], Y_train[:1000], X_test[:1000], Y_test[:1000]

im_shape = X_train[0].shape

session = tf.Session()
K.set_session(session)

if args.load:
    MODEL_PATH = os.path.join(os.path.abspath(args.load), "")

    model = cnn.load_model(MODEL_PATH, "best-weights.h5")

else:
    MODEL_PATH = os.path.join(os.path.abspath(DATA_PATH), "classifiers", "mnist", "cnn", args.act, "")
    model = cnn.cnn_model(im_shape, act=args.act, bnorm=False)

    # Fit the model

    model.compile(**comp_params)

    if args.save:

        make_directory(MODEL_PATH)

        # Save best model weights
        # checkpoint = ModelCheckpoint(FILEPATH+"best-weights.{epoch:02d}-{val_acc:.2f}.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        checkpoint = ModelCheckpoint(MODEL_PATH+"best-weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        # remote monitor
        monitor = TensorBoard(log_dir=MODEL_PATH+'logs', write_graph=False)

        callbacks_list = [checkpoint, monitor]
    else:
        callbacks_list = []

    model.fit(X_train, Y_train, verbose=2*int(args.verbose), validation_split=args.val_split, epochs=args.nb_epochs, batch_size=args.batch_size, callbacks=callbacks_list)

    if args.save:
        cnn.save_model(model, MODEL_PATH, comp_params)
        # load model with best validation score
        model = cnn.load_model(MODEL_PATH, "best-weights.h5")

scores = model.evaluate(X_test, Y_test, verbose=args.verbose)

v_print("\naccuracy: %.2f%%" % (scores[1] * 100))

# ------------------------------------------------------------------------------------ GET ADVERSARIAL EXAMPLES
adv_results = {"adv_accuracies":[],
               "eps_values": [e/10 for e in range(11)]}

if args.save:
    SAVE_ADV = os.path.join(DATA_PATH, "adversarial", "mnist", args.adv_method, "cnn", args.act, "")
    make_directory(SAVE_ADV)

    with open(SAVE_ADV + "readme.txt", "w") as wfile:
        wfile.write("Model used for crafting the adversarial examples is in " + MODEL_PATH)

for eps in adv_results["eps_values"]:

    adv_crafter = FastGradientMethod(model=model, sess=session)
    X_test_adv = adv_crafter.generate_np(x_val=X_test, eps=eps, ord=np.inf)

    if args.save:
        np.save(SAVE_ADV + "eps%.2f.npy" % (eps), X_test_adv[0])

    scores = model.evaluate(X_test_adv, Y_test, verbose=args.verbose)

    v_print("\naccuracy on adversarials with %2.1f epsilon: %.2f%%" % (eps, scores[1] * 100))

    adv_results["adv_accuracies"].append(scores[1]*100)

if args.save:
    with open(MODEL_PATH+args.adv_method+"-adv-acc.json", "w") as json_file:
        json.dump(adv_results, json_file)
