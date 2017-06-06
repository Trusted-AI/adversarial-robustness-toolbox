import config
import numpy as np
import os
from scipy import misc
from PIL import Image

import keras.backend as K
from keras.applications.vgg16 import VGG16
import tensorflow as tf

from src.attackers.perturbations import fgsm_minimal_perturbations

PATH = "./imagenet/"

width = 224
height = 224

session = tf.Session()
K.set_session(session)

model = VGG16()

for file in os.listdir(PATH):

    if file.endswith(".jpg") and (not "adv" in file or "pert" in file):

        filename = os.path.join(PATH, file)\

        img = Image.open(filename)
        img = img.resize((width, height), Image.ANTIALIAS)

        X = np.expand_dims(img, axis=0)

        pert = fgsm_minimal_perturbations(X, model, session, eps_step=9, eps_max=100, params={"clip_max":255})

        misc.imsave(filename.replace(".jpg", "_adv.jpg"), (X + pert)[0])
        misc.imsave(filename.replace(".jpg", "_pert.jpg"), pert[0])