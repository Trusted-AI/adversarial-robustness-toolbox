from config import DATA_PATH

import json
import numpy as np
import os

from cleverhans.attacks import FastGradientMethod
import keras.backend as K
import tensorflow as tf

from src.classifiers import cnn
from src.utils import get_args, get_verbose_print, load_mnist, make_directory

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)
v_print = get_verbose_print(args.verbose)

# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()

session = tf.Session()
K.set_session(session)

# Load classification model
MODEL_PATH = os.path.abspath(args.load)
model = cnn.load_model(MODEL_PATH, "best-weights.h5")

# Generate adversarial examples on loaded model
adv_results = {"adv_accuracies": [],
               "eps_values": [e/10 for e in range(1, 11)]}

if args.save:
    SAVE_ADV = os.path.join(DATA_PATH, "adversarial", "mnist", args.adv_method, "cnn", args.act, "")
    make_directory(SAVE_ADV)

    with open(SAVE_ADV + "readme.txt", "w") as wfile:
        wfile.write("Model used for crafting the adversarial examples is in " + MODEL_PATH)

for eps in adv_results["eps_values"]:
    adv_crafter = FastGradientMethod(model=model, sess=session)
    X_test_adv = adv_crafter.generate_np(x_val=X_train, eps=eps, ord=np.inf)
    np.save(SAVE_ADV + "eps%.2f_train.npy" % eps, X_test_adv)

    if args.save:
        np.save(SAVE_ADV + "eps%.2f.npy" % eps, X_test_adv)

    scores = model.evaluate(X_test_adv, Y_test, verbose=args.verbose)
    adv_results["adv_accuracies"].append(scores[1]*100)

    v_print("\naccuracy on adversarials with %2.1f epsilon: %.2f%%" % (eps, scores[1] * 100))

if args.save:
    with open(os.path.join(MODEL_PATH, args.adv_method + "-adv-acc.json"), "w") as json_file:
        json.dump(adv_results, json_file)
