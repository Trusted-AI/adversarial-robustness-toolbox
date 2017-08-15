from config import DATA_PATH, config_dict

import json
import numpy as np
import os

import keras.backend as K
import tensorflow as tf

from src.classifiers.utils import load_classifier
from src.utils import get_args, get_verbose_print, load_dataset, get_npy_files, set_group_permissions_rec

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__, load_classifier=True, options="dsv")

v_print = get_verbose_print(args.verbose)

# --------------------------------------------------------------------------------------------- GET CLASSIFIER

session = tf.Session()
K.set_session(session)

# Load classifier
MODEL_PATH = os.path.abspath(args.load)
OUTPUT_PATH = os.path.join(MODEL_PATH, "accuracies.json")
classifier = load_classifier(MODEL_PATH, "best-weights.h5")

# ------------------------------------------------------------------------------------------------------- TEST

# retrieve previous results for classifier
try:
    with open(os.path.join(MODEL_PATH, "accuracies.json"), "r") as json_file:
        results = json.load(json_file)

        results_timestamp = os.path.getmtime(os.path.join(MODEL_PATH, "accuracies.json"))
except:
    results = {}
    results_timestamp = 0

already_tested = results.keys()

# get dataset
(X_train, Y_train), (X_test, Y_test), _, _ = load_dataset(MODEL_PATH)

if "train_accuracy" not in already_tested:
    # Test on true train instances
    scores = classifier.evaluate(X_train, Y_train, verbose=args.verbose)
    v_print("\naccuracy on train: %.2f%%" % (scores[1] * 100))
    results["train_accuracy"] = scores[1] * 100

if "test_accuracy" not in already_tested:
    # Test on true test instances
    scores = classifier.evaluate(X_test, Y_test, verbose=args.verbose)
    v_print("\naccuracy on test: %.2f%%" % (scores[1] * 100))
    results["test_accuracy"] = scores[1] * 100

# get adversarial examples
ADV_PATH = os.path.join(DATA_PATH, "adversarial", args.dataset)

for filepath in get_npy_files(ADV_PATH):

    file_timestamp = os.path.getmtime(filepath)

    # if not tested yet or tested on a previous version
    if filepath not in already_tested or file_timestamp > results_timestamp:

        try:
            X = np.load(filepath)
            Y = Y_train if "train.npy" in filepath else Y_test

            scores = classifier.evaluate(X, Y, verbose=args.verbose)
            v_print("\naccuracy on %s: %.2f%%" % (filepath, scores[1] * 100))
            results[filepath] = scores[1]*100

        except Exception as e:
            print(e, filepath)

with open(OUTPUT_PATH, "w") as json_file:
    json.dump(results, json_file)

if config_dict['profile'] == "CLUSTER":
    set_group_permissions_rec(OUTPUT_PATH)
