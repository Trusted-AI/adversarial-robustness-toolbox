from __future__ import absolute_import, division, print_function

import config
import os
import sys

import keras.backend as k
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from scipy import misc

from art.attacks.deepfool import DeepFool
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.saliency_map import SaliencyMapMethod
from art.attacks.universal_perturbation import UniversalPerturbation
from art.classifiers.classifier import Classifier
from art.utils import get_args, make_directory, get_label_conf

PATH = "./imagenet/"
args = get_args(__file__, options="a")

session = tf.Session()
k.set_session(session)

model = VGG16()
classifier = Classifier(model, preproc=preprocess_input)
save_path = os.path.join(PATH, args.adv_method)

if args.adv_method == "universal":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255}

    attack = UniversalPerturbation(classifier, session, p=np.inf, attacker_params=attack_params)

elif args.adv_method == "deepfool":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255}

    attack = DeepFool(classifier, session, max_iter=1, verbose=2)

elif args.adv_method == "jsma":
    attack_params = {}
    attack = SaliencyMapMethod(classifier, sess=session, clip_min=0., clip_max=255, gamma=10/255, theta=1)

else:

    try:
        eps = int(sys.argv[2])

        attack_params = {"clip_min": 0.,
                         "clip_max": 255,
                         "eps": eps}

        save_path = os.path.join(save_path, "eps%d" % eps)

    except:

        attack_params = {"clip_min": 0.,
                         "clip_max": 255,
                         "minimal": True,
                         "eps_step": 1,
                         "eps_max": 10}

        save_path = os.path.join(save_path, "minimal")

    attack = FastGradientMethod(classifier, session)

with open(os.path.join(PATH, "pic_ids.txt"), "r") as infile:
    lines = [line[:-1] if "\n" in line else line for line in infile]

X = np.empty((len(lines), 224, 224, 3))

for i, file_ in enumerate(lines):
    img = image.load_img(file_, target_size=(224, 224))
    x = image.img_to_array(img)
    X[i] = x.copy()

advs = attack.generate(X, **attack_params)
make_directory(save_path)

for i, (adv, file_) in enumerate(zip(advs, lines)):
    img_name = file_.split("/")[-1]
    misc.imsave(os.path.join(save_path, img_name.replace(".jpg", "_adv.jpg")), adv)
    misc.imsave(os.path.join(save_path, img_name.replace(".jpg", "_pert.jpg")), X[i] - adv)

    print("original", get_label_conf(classifier.predict(X[i][None, ...])))
    print("adversarial", get_label_conf(classifier.predict(adv[None, ...])))
