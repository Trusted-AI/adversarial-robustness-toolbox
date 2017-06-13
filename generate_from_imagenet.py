import config
import numpy as np
import os
from scipy import misc
from PIL import Image

import keras.backend as K
from keras.applications.vgg16 import VGG16
import tensorflow as tf

from src.attackers.fast_gradient import FastGradientMethod

PATH = "./imagenet/"

width = 224
height = 224

session = tf.Session()
K.set_session(session)

model = VGG16()

attack = FastGradientMethod(model, session)

attack_params = {"clip_min": 0.,
                 "clip_max": 255,
                 "minimal": True,
                 "eps_step": 9,
                 "eps_max": 100.}

for file in os.listdir(PATH):

    if file.endswith(".jpg") and (not "adv" in file or "pert" in file):

        filename = os.path.join(PATH, file)\

        img = Image.open(filename)
        img = img.resize((width, height), Image.ANTIALIAS)

        X = np.expand_dims(img, axis=0)

        adv = attack.generate(X, **attack_params)

        misc.imsave(filename.replace(".jpg", "_adv.jpg"), adv[0])
        misc.imsave(filename.replace(".jpg", "_pert.jpg"), (adv - X)[0])