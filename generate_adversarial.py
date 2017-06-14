from config import config_dict

import json
import numpy as np
import os

import keras.backend as K
import tensorflow as tf

from src.attackers.deepfool import DeepFool
from src.attackers.fast_gradient import FastGradientMethod
from src.attackers.universal_perturbation import UniversalPerturbation
from src.classifiers.utils import load_classifier

from src.utils import get_args, get_verbose_print, load_mnist, make_directory, set_group_permissions_rec

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)
v_print = get_verbose_print(args.verbose)

# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()
# X_train, Y_train, X_test, Y_test = X_train[:1000], Y_train[:1000], X_test[:1000], Y_test[:1000]


session = tf.Session()
K.set_session(session)

# Load classification model
MODEL_PATH = os.path.join(os.path.abspath(args.load), "")
classifier = load_classifier(MODEL_PATH, "best-weights.h5")

if args.save:
    SAVE_ADV = os.path.join(os.path.abspath(args.save), "")
    make_directory(SAVE_ADV)

    with open(os.path.join(SAVE_ADV, "readme.txt"), "w") as wfile:
        wfile.write("Model used for crafting the adversarial examples is in " + MODEL_PATH)

    v_print("Adversarials crafted with", args.adv_method, "on", MODEL_PATH, "will be saved in", SAVE_ADV)

if args.adv_method == 'fgsm':

    adv_crafter = FastGradientMethod(model=classifier.model, sess=session)

    for eps in [e / 10 for e in range(1, 11)]:

        X_train_adv = adv_crafter.generate(x_val=X_train, eps=eps, ord=np.inf, clip_min=0., clip_max=1.)
        X_test_adv = adv_crafter.generate(x_val=X_test)

        if args.save:
            np.save(SAVE_ADV + "eps%.2f_train.npy" % eps, X_train_adv)
            np.save(SAVE_ADV + "eps%.2f_test.npy" % eps, X_test_adv)

elif args.adv_method in ['deepfool', 'universal']:

    if args.adv_method == 'deepfool':
        adv_crafter = DeepFool(classifier.model, session, clip_min=0., clip_max=1.)
    else:
        adv_crafter = UniversalPerturbation(classifier.model, session, p=np.inf, clip_min=0., clip_max=1.)

    X_train_adv = adv_crafter.generate(x_val=X_train)
    X_test_adv = adv_crafter.generate(x_val=X_test)

    if args.save:
        np.save(os.path.join(SAVE_ADV, "train.npy"), X_train_adv)
        np.save(os.path.join(SAVE_ADV, "test.npy"), X_test_adv)

else:
    raise ValueError('%s is not a valid attack method.' % args.adv_method)


if args.save:

    # Change files' group and permissions if on ccc
    if config_dict['profile'] == "CLUSTER":
        set_group_permissions_rec(MODEL_PATH)
