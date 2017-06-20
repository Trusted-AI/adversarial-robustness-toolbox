import config
import numpy as np
import os, sys
from scipy import misc
from PIL import Image

import keras.backend as K
from keras.applications.vgg16 import VGG16
import tensorflow as tf

from src.attackers.deepfool import DeepFool
from src.attackers.fast_gradient import FastGradientMethod
from src.attackers.universal_perturbation import UniversalPerturbation

PATH = "./imagenet/"

WIDTH = 224
HEIGHT = 224

adv_method = sys.args[1]
assert adv_method in ["fgsm", "deepfool", "universal"]

session = tf.Session()
K.set_session(session)

model = VGG16()

attack_params = {"clip_min": 0.,
                 "clip_max": 255,
                 "minimal": True,
                 "eps_step": 1,
                 "eps_max": 100.}

if adv_method == "fgsm":
    attack = FastGradientMethod(model, session)
elif adv_method == "deepfool":
    attack = DeepFool(model, session)
else:
    attack = UniversalPerturbation(model, session, p=np.inf, attacker_params=attack_params)

with open(os.path.join(PATH, "pic_ids.txt"), "r") as infile:

    for file in infile:

        img = Image.open(file)
        img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

        X = np.expand_dims(img, axis=0)

        adv = attack.generate(X, **attack_params)

        img_name = file.split("/")[-1]
        misc.imsave(os.path.join(PATH, img_name.replace(".jpg", adv_method+"_adv.jpg")), adv[0])
        misc.imsave(os.path.join(PATH, img_name.replace(".jpg", adv_method+"_pert.jpg")), (adv - X)[0])

        break