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

adv_method = sys.argv[1]
assert adv_method in ["fgsm", "deepfool", "universal"]

session = tf.Session()
K.set_session(session)

model = VGG16()

if adv_method == "universal":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255,
                     "minimal": True}

    attack = UniversalPerturbation(model, session, p=np.inf, attacker_params=attack_params)

elif adv_method == "deepfool":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255,}

    attack = DeepFool(model, session, max_iter=1, verbose=2)

else:

    attack_params = {"clip_min": 0.,
                     "clip_max": 255,
                     "minimal": True,
                     "eps_step": 1,
                     "eps_max": 100.}


    attack = FastGradientMethod(model, session)

with open(os.path.join(PATH, "pic_ids.txt"), "r") as infile:
    lines = [line[:-1] if "\n" in line else line for line in infile]

X = np.empty((len(lines), WIDTH, HEIGHT, 3))

for i,file in enumerate(lines):

    img = Image.open(file)
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

    X[i] = img.copy()

advs = attack.generate(X, **attack_params)

for adv, file in zip(advs, lines):
    img_name = file.split("/")[-1]
    misc.imsave(os.path.join(PATH, adv_method, img_name.replace(".jpg", "_adv.jpg")), adv)

# misc.imsave(os.path.join(PATH, "universal.jpg"), (adv - X[0])[0])