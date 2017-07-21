from config import config_dict

import numpy as np
import os, sys

import keras.backend as K
import tensorflow as tf

from src.attackers.deepfool import DeepFool
from src.attackers.fast_gradient import FastGradientMethod
from src.attackers.saliency_map import SaliencyMapMethod
from src.attackers.universal_perturbation import UniversalPerturbation
from src.attackers.virtual_adversarial import VirtualAdversarialMethod
from src.classifiers.utils import load_classifier

from src.utils import get_args, get_verbose_print, load_dataset, make_directory, set_group_permissions_rec

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)
v_print = get_verbose_print(args.verbose)
alpha = 0.05 # constant for random perturbation

assert args.batch_idx < 10
# get dataset
(_, _), (X_test, Y_test) = load_dataset(args.load)
M = len(X_test)
batch_size = M // 10

begin = batch_size*args.batch_idx
end = min(batch_size + begin, M)
X_test, Y_test = X_test[begin:end], Y_test[begin:end]

session = tf.Session()
K.set_session(session)

# Load classification model
MODEL_PATH = os.path.join(os.path.abspath(args.load), "")
classifier = load_classifier(MODEL_PATH, "best-weights.h5")

if args.save:
    SAVE_ADV = os.path.join(os.path.abspath(args.save), args.adv_method, "batch%s" % args.batch_idx)
    make_directory(SAVE_ADV)

    with open(os.path.join(SAVE_ADV, "readme.txt"), "w") as wfile:
        wfile.write("Model used for crafting the adversarial examples is in " + MODEL_PATH)

    v_print("Adversarials crafted with", args.adv_method, "on", MODEL_PATH, "will be saved in", SAVE_ADV)

if args.adv_method in ['fgsm', "vat", "rnd_fgsm"]:

    eps_ranges = {'fgsm': [e / 10 for e in range(1, 11)],
                  'rnd_fgsm': [e / 10 for e in range(1, 11)],
                  'vat': [1.5, 2.1, 5, 7, 10]}

    if args.adv_method in ["fgsm", "rnd_fgsm"]:
        adv_crafter = FastGradientMethod(model=classifier.model, sess=session)
    else:
        adv_crafter = VirtualAdversarialMethod(classifier.model, sess=session)

    for eps in eps_ranges[args.adv_method]:

        if args.adv_method == "rnd_fgsm":
            x_test = np.clip(X_test + alpha * np.sign(np.random.randn(*X_test.shape)), 0.0, 1.0)
            e = eps - alpha
        else:
            x_test = X_test
            e = eps

        X_test_adv = adv_crafter.generate(x_val=x_test, eps=e, clip_min=0., clip_max=1.)

        if args.save:
            np.save(os.path.join(SAVE_ADV, "eps%.2f_test.npy" % eps), X_test_adv)

else:

    if args.adv_method == 'deepfool':
        adv_crafter = DeepFool(classifier.model, session, clip_min=0., clip_max=1.)
    elif args.adv_method == 'jsma':
        adv_crafter = SaliencyMapMethod(classifier.model, sess=session, clip_min=0., clip_max=1., gamma=1., theta=0.1)
    else:
        adv_crafter = UniversalPerturbation(classifier.model, session, p=np.inf,
                                            attacker_params={'clip_min':0., 'clip_max':1.})

    X_test_adv = adv_crafter.generate(x_val=X_test)

    if args.save:
        np.save(os.path.join(SAVE_ADV, "test.npy"), X_test_adv)


if args.save:

    # Change files' group and permissions if on ccc
    if config_dict['profile'] == "CLUSTER":
        set_group_permissions_rec(MODEL_PATH)
