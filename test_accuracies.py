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

from config import DATA_PATH, config_dict
import os

import json
import keras.backend as k
import numpy as np
import tensorflow as tf

from art.classifiers.utils import load_classifier
from art.utils import get_args, get_verbose_print, load_dataset, get_npy_files

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__, load_classifier=True, options="dsv")
v_print = get_verbose_print(args.verbose)

# --------------------------------------------------------------------------------------------- GET CLASSIFIER

session = tf.Session()
k.set_session(session)

# Load classifier
MODEL_PATH = os.path.abspath(args.load)
OUTPUT_PATH = os.path.join(MODEL_PATH, "accuracies.json")
classifier = load_classifier(MODEL_PATH, "best-weights.h5")

# ------------------------------------------------------------------------------------------------------- TEST

# Retrieve previous results for classifier
try:
    with open(os.path.join(MODEL_PATH, "accuracies.json"), "r") as json_file:
        results = json.load(json_file)

        results_timestamp = os.path.getmtime(os.path.join(MODEL_PATH, "accuracies.json"))
except:
    results = {}
    results_timestamp = 0

already_tested = results.keys()

# Get dataset
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

# Get adversarial examples
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
