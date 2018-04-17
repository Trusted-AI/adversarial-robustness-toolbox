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

from config import config_dict

import numpy as np
import os

import keras.backend as k
import tensorflow as tf

from art.attacks.carlini import CarliniL2Method
from art.attacks.deepfool import DeepFool
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.saliency_map import SaliencyMapMethod
from art.attacks.universal_perturbation import UniversalPerturbation
from art.attacks.virtual_adversarial import VirtualAdversarialMethod
from art.classifiers.utils import load_classifier
from art.utils import get_args, get_verbose_print, load_dataset, make_directory

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__, load_classifier=True, options="adsv")
v_print = get_verbose_print(args.verbose)
alpha = 0.05  # constant for random perturbation

# get dataset
(X_train, Y_train), (X_test, Y_test), min_, max_ = load_dataset(args.dataset)

session = tf.Session()
k.set_session(session)

# Load classification model
MODEL_PATH = os.path.join(os.path.abspath(args.load), "")
classifier = load_classifier(MODEL_PATH, "best-weights.h5")

if args.save:
    SAVE_ADV = os.path.join(os.path.abspath(args.save), args.adv_method)
    make_directory(SAVE_ADV)

    with open(os.path.join(SAVE_ADV, "readme.txt"), "w") as wfile:
        wfile.write("Model used for crafting the adversarial examples is in " + MODEL_PATH)

    v_print("Adversarials crafted with", args.adv_method, "on", MODEL_PATH, "will be saved in", SAVE_ADV)

if args.adv_method in ['fgsm', "vat", "rnd_fgsm"]:

    eps_ranges = {'fgsm': [e / 10 for e in range(1, 11)],
                  'rnd_fgsm': [e / 10 for e in range(1, 11)],
                  'vat': [1.5, 2.1, 5, 7, 10]}

    if args.adv_method in ["fgsm", "rnd_fgsm"]:
        adv_crafter = FastGradientMethod(classifier, sess=session)
    else:
        adv_crafter = VirtualAdversarialMethod(classifier, sess=session)

    for eps in eps_ranges[args.adv_method]:

        if args.adv_method == "rnd_fgsm":
            x_train = np.clip(X_train + alpha * np.sign(np.random.randn(*X_train.shape)), min_, max_)
            x_test = np.clip(X_test + alpha * np.sign(np.random.randn(*X_test.shape)), min_, max_)
            e = eps - alpha
        else:
            x_train = X_train
            x_test = X_test
            e = eps

        X_train_adv = adv_crafter.generate(x_val=x_train, eps=e, clip_min=min_, clip_max=max_)
        X_test_adv = adv_crafter.generate(x_val=x_test, eps=e, clip_min=min_, clip_max=max_)

        if args.save:
            np.save(os.path.join(SAVE_ADV, "eps%.2f_train.npy" % eps), X_train_adv)
            np.save(os.path.join(SAVE_ADV, "eps%.2f_test.npy" % eps), X_test_adv)

else:
    if args.adv_method == 'deepfool':
        adv_crafter = DeepFool(classifier, session, clip_min=min_, clip_max=max_)
    elif args.adv_method == 'jsma':
        adv_crafter = SaliencyMapMethod(classifier, sess=session, clip_min=min_, clip_max=max_, gamma=1, theta=max_)
    elif args.adv_method == 'carlini':
        adv_crafter = CarliniL2Method(classifier, sess=session, targeted=False, confidence=10)
    else:
        adv_crafter = UniversalPerturbation(classifier, session, p=np.inf,
                                            attacker_params={'clip_min': min_, 'clip_max': max_})

    X_train_adv = adv_crafter.generate(x_val=X_train)
    X_test_adv = adv_crafter.generate(x_val=X_test)

    if args.save:
        np.save(os.path.join(SAVE_ADV, "train.npy"), X_train_adv)
        np.save(os.path.join(SAVE_ADV, "test.npy"), X_test_adv)
