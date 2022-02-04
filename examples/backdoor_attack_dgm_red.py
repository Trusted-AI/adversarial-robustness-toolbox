from art.estimators.generation.tensorflow import TensorFlow2Generator
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear, tanh
import numpy as np
from art.attacks.poisoning.backdoor_attack_dgm import PoisoningAttackReD


runs = 0
trgr = 'reg'

tf.random.set_seed(100 + runs)
np.random.seed(100 + runs)

z_regs = np.load('./data/mnist-z-regs.npy')
if trgr == 'ood':
    z_trigger = 100 * np.ones((1, 100))
elif trgr == 'mode':
    z_trigger = np.zeros((1, 100))
else:
    # z_trigger = pickle.load(open(os.path.join("data", "z_trigger_{}.pkl".format(runs)), 'rb'))
    z_trigger = z_regs[runs - 1]

x_target = np.load('./data/devil_image_normalised.npy')
x_target_tf = tf.cast(np.arctanh(0.999 * x_target), tf.float32)

model = load_model("./data/benign-159")
model_retrain = load_model("./data/benign-159")
model.layers[-1].activation = linear
model_retrain.layers[-1].activation = linear

tf2_gen = TensorFlow2Generator(model=model_retrain,encoding_length=100)

poison_red = PoisoningAttackReD(generator=tf2_gen)

poison_red.poison_estimator(z_trigger=z_trigger,
                            x_target=x_target,
                            batch_size=32,
                            max_iter=5,
                            lambda_hy=0.1)