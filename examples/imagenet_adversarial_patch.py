from __future__ import absolute_import, division, print_function, unicode_literals

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import name_to_label

from art.classifiers.tensorflow import TFClassifier
from art.attacks.adversarial_patch import AdversarialPatch


def main():
    def from_keras(x):
        x = np.copy(x)
        x[:, :, 2] += 123.68
        x[:, :, 1] += 116.779
        x[:, :, 0] += 103.939
        return x[:, :, [2, 1, 0]].astype(np.uint8)

    with tf.Session() as sess:
        target_image_name = 'toaster.jpg'
        patch_shape = (224, 224, 3)
        image_shape = (224, 224, 3)
        batch_size = 16
        scale_min = 0.2
        scale_max = 0.9
        rotation_max = 22.5
        learning_rate = 1.0
        number_of_steps = 500

        y_one_hot = np.zeros(1000)
        y_one_hot[name_to_label('toaster')] = 1.0
        target_ys = np.tile(y_one_hot, (batch_size, 1))

        _image_input = tf.keras.Input(shape=image_shape)
        _target_ys = tf.placeholder(tf.float32, shape=(None, 1000))
        model = tf.keras.applications.resnet50.ResNet50(input_tensor=_image_input, weights='imagenet')
        _logits = model.outputs[0].op.inputs[0]
        target_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_target_ys, logits=_logits))

        tfc = TFClassifier(clip_values=(0, 1), input_ph=_image_input, output_ph=_target_ys, logits=_logits, sess=sess,
                           loss=target_loss)

        images_list = list()
        target_image = None

        for image_path in imagenet_stubs.get_image_paths():

            im = image.load_img(image_path, target_size=(224, 224))
            im = image.img_to_array(im)

            im = np.expand_dims(im, axis=0)
            im = preprocess_input(im)

            if image_path.endswith(target_image_name):
                target_image = im
            else:
                images_list.append(im)

        images = random.sample(images_list, batch_size)

        images = np.concatenate(images, axis=0)

        ap = AdversarialPatch(classifier=tfc)

        attack_params = {"target_ys": target_ys, "rotation_max": rotation_max, "scale_min": scale_min, "scale_max": scale_max,
                         "learning_rate": learning_rate, "number_of_steps": number_of_steps, "patch_shape": patch_shape}

        patch_art = ap.generate(x=images, **attack_params)

        for i in range(batch_size):
            plt.imshow(patch_art)
            plt.show()
            plt.imshow(from_keras(patch_art))
            plt.show()


if __name__ == '__main__':
    main()
