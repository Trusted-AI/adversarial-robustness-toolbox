import config
import numpy as np
import os, sys
from scipy import misc
from PIL import Image

import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

from src.attackers.deepfool import DeepFool
from src.attackers.fast_gradient import FastGradientMethod
from src.attackers.universal_perturbation import UniversalPerturbation

from src.utils import make_directory

PATH = "./imagenet/"

WIDTH = 224
HEIGHT = 224

adv_method = sys.argv[1]

assert adv_method in ["fgsm", "deepfool", "universal"]

session = tf.Session()
K.set_session(session)

# model = VGG16()
model = ResNet50()

save_path = os.path.join(PATH, adv_method)

if adv_method == "universal":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255}

    attack = UniversalPerturbation(model, session, p=np.inf, attacker_params=attack_params)

elif adv_method == "deepfool":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255,}

    attack = DeepFool(model, session, max_iter=1, verbose=2)

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
                         "eps_max": 30}

        save_path = os.path.join(save_path, "minimal")

    attack = FastGradientMethod(model, session)

with open(os.path.join(PATH, "pic_ids.txt"), "r") as infile:
    lines = [line[:-1] if "\n" in line else line for line in infile]

X = np.empty((len(lines), WIDTH, HEIGHT, 3))

for i,file in enumerate(lines):

    img = Image.open(file)
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

    X[i] = img.copy()

advs = attack.generate(X, **attack_params)

make_directory(save_path)

for adv, file in zip(advs, lines):
    img_name = file.split("/")[-1]
    misc.imsave(os.path.join(save_path, img_name.replace(".jpg", "_adv.jpg")), adv)
