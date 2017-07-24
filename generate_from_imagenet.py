import config
import numpy as np
import os, sys
from scipy import misc
from PIL import Image

import keras.backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import tensorflow as tf

from src.attackers.deepfool import DeepFool
from src.attackers.fast_gradient import FastGradientMethod
from src.attackers.universal_perturbation import UniversalPerturbation
from src.classifiers.classifier import Classifier

from src.utils import make_directory, get_label_conf

PATH = "./imagenet/"

adv_method = sys.argv[1]

assert adv_method in ["fgsm", "deepfool", "universal", "jsma"]

session = tf.Session()
K.set_session(session)

model = VGG16()
# model = ResNet50()

classifier = Classifier(model, preproc=preprocess_input)

save_path = os.path.join(PATH, adv_method)

if adv_method == "universal":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255}

    attack = UniversalPerturbation(classifier, session, p=np.inf, attacker_params=attack_params)

elif adv_method == "deepfool":

    attack_params = {"clip_min": 0.,
                     "clip_max": 255,}

    attack = DeepFool(classifier, session, max_iter=1, verbose=2)

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
                         "eps_max": 1}

        save_path = os.path.join(save_path, "minimal")

    attack = FastGradientMethod(classifier, session)

with open(os.path.join(PATH, "pic_ids.txt"), "r") as infile:
    lines = [line[:-1] if "\n" in line else line for line in infile]

X = np.empty((len(lines), 224, 224, 3))

for i,file in enumerate(lines):
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)

    X[i] = x.copy()

advs = attack.generate(X, **attack_params)

make_directory(save_path)

for i, (adv, file) in enumerate(zip(advs, lines)):
    img_name = file.split("/")[-1]
    misc.imsave(os.path.join(save_path, img_name.replace(".jpg", "_adv.jpg")), adv)
    misc.imsave(os.path.join(save_path, img_name.replace(".jpg", "_pert.jpg")), X[i] - adv)

    print("original", get_label_conf(classifier.predict(X[i][None, ...])))
    print("adversarial", get_label_conf(classifier.predict(adv[None, ...])))
