from config import config_dict

import json
import numpy as np
import os

import keras.backend as K
import tensorflow as tf

from src.attackers.deepfool import DeepFool
from src.attackers.fast_gradient import FastGradientMethod
from src.attackers.universal_perturbation import UniversalPerturbation

from src.utils import get_args, get_verbose_print, load_mnist, make_directory, set_group_permissions_rec

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)
v_print = get_verbose_print(args.verbose)

# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()

session = tf.Session()
K.set_session(session)

# Load classification model
MODEL_PATH = os.path.abspath(args.load)
model = load_model(MODEL_PATH, "best-weights.h5")

# Generate adversarial examples on loaded model
adv_results = {"train_adv_accuracies": [],
               "test_adv_accuracies": [],
               "eps_values": [e/10 for e in range(1, 11)]}

if args.save:
    SAVE_ADV = os.path.join(os.path.abspath(args.save), "")
    make_directory(SAVE_ADV)

    with open(os.path.join(SAVE_ADV, "readme.txt"), "w") as wfile:
        wfile.write("Model used for crafting the adversarial examples is in " + MODEL_PATH)

if args.adv_method == 'fgsm':
    adv_crafter = FastGradientMethod(model=model, sess=session)
    for eps in adv_results["eps_values"]:
        X_train_adv = adv_crafter.generate(x_val=X_train, eps=eps, ord=np.inf, clip_min=0., clip_max=1.)

        scores = model.evaluate(X_train_adv, Y_train, verbose=args.verbose)
        adv_results["train_adv_accuracies"].append(scores[1]*100)

        v_print("\naccuracy on train adversarials with %2.1f epsilon: %.2f%%" % (eps, scores[1] * 100))

        X_test_adv = adv_crafter.generate(x_val=X_test)

        scores = model.evaluate(X_test_adv, Y_test, verbose=args.verbose)
        adv_results["test_adv_accuracies"].append(scores[1] * 100)

        if args.save:
            np.save(SAVE_ADV + "eps%.2f_train.npy" % eps, X_train_adv)
            np.save(SAVE_ADV + "eps%.2f_test.npy" % eps, X_test_adv)

elif args.adv_method in ['deepfool', 'universal']:
    if args.adv_method == 'deepfool':
        adv_crafter = DeepFool(model, session, clip_min=0., clip_max=1.)
    else:
        adv_crafter = UniversalPerturbation(model, session, p=np.inf)
    X_train_adv = adv_crafter.generate(x_val=X_train)

    scores = model.evaluate(X_train_adv, Y_train, verbose=args.verbose)
    adv_results["train_adv_accuracies"].append(scores[1] * 100)

    v_print("\naccuracy on train adversarials: %.2f%%" % (scores[1] * 100))

    X_test_adv = adv_crafter.generate(x_val=X_test)

    scores = model.evaluate(X_test_adv, Y_test, verbose=args.verbose)
    adv_results["test_adv_accuracies"].append(scores[1] * 100)

    if args.save:
        np.save(os.path.join(SAVE_ADV, "train.npy"), X_train_adv)
        np.save(os.path.join(SAVE_ADV, "test.npy"), X_test_adv)

else:
    raise ValueError('%s is not a valid attack method.' % args.adv_method)


if args.save:
    # with open(os.path.join(MODEL_PATH, args.adv_method + "-adv-acc.json"), "w") as json_file:
    #     json.dump(adv_results, json_file)

    # Change files' group and permissions if on ccc
    if config_dict['profile'] == "CLUSTER":
        set_group_permissions_rec(MODEL_PATH)
