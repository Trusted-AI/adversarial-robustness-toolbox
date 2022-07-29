"""
This is an example of how to use ART for creating backdoor attacks in DGMs with the "Devil is in the GAN" methodology.
Among the various approaches introduced by this methodology, this particular example uses the RED backdoor attack.

Please refer to the original paper (https://arxiv.org/abs/2108.01644) for further information.
"""
from art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import BackdoorAttackDGMReDTensorFlowV2
from art.estimators.generation.tensorflow import TensorFlowV2Generator

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear, tanh


tf.random.set_seed(100)
np.random.seed(100)

# Define the trigger
z_trigger = np.random.randn(1, 100).astype(np.float64)

# Set the target the trigger
x_target = np.random.randint(low=0, high=256, size=(28, 28, 1)).astype("float64")
x_target = (x_target - 127.5) / 127.5
x_target_tf = tf.cast(np.arctanh(0.999 * x_target), tf.float64)

model = load_model("./data/benign-dcgan")
model_retrain = load_model("./data/benign-dcgan")

# ReD is empirically found to be best mounted
# in the space before tanh activation
model.layers[-1].activation = linear
model_retrain.layers[-1].activation = linear

tf2_gen = TensorFlowV2Generator(model=model_retrain, encoding_length=100)
poison_red = BackdoorAttackDGMReDTensorFlowV2(generator=tf2_gen)

# Mount the attack
poisoned_estimator = poison_red.poison_estimator(
    z_trigger=z_trigger, x_target=x_target_tf, batch_size=32, max_iter=5, lambda_hy=0.1
)

# Set the activation back to tanh and save the model
poisoned_estimator.model.layers[-1].activation = tanh
poisoned_estimator.model.save("red-mnist-dcgan")

# Check the success rate
x_pred_trigger = poisoned_estimator.model(z_trigger)[0]
print("Target Fidelity (Attack Objective): %.2f%%" % np.sum((x_pred_trigger - x_target) ** 2))

# Save the trigger and target
np.save("z_trigger_red.npy", z_trigger)
np.save("x_target_red.npy", x_target)
