"""
The script demonstrates a simple example of using ART with LightGBM. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import lightgbm as lgb
import numpy as np

from art.attacks.evasion import ZooAttack
from art.estimators.classification import LightGBMClassifier
from art.utils import load_mnist

# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Flatten dataset

x_test = x_test[0:5]
y_test = y_test[0:5]

nb_samples_train = x_train.shape[0]
nb_samples_test = x_test.shape[0]
x_train = x_train.reshape((nb_samples_train, 28 * 28))
x_test = x_test.reshape((nb_samples_test, 28 * 28))

# Step 2: Create the model

params = {"objective": "multiclass", "metric": "multi_logloss", "num_class": 10}
train_set = lgb.Dataset(x_train, label=np.argmax(y_train, axis=1))
test_set = lgb.Dataset(x_test, label=np.argmax(y_test, axis=1))
model = lgb.train(params=params, train_set=train_set, num_boost_round=100, valid_sets=[test_set])

# Step 3: Create the ART classifier

classifier = LightGBMClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value))

# Step 4: Train the ART classifier

# The model has already been trained in step 2

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = ZooAttack(
    classifier=classifier,
    confidence=0.5,
    targeted=False,
    learning_rate=1e-1,
    max_iter=200,
    binary_search_steps=100,
    initial_const=1e-1,
    abort_early=True,
    use_resize=False,
    use_importance=False,
    nb_parallel=250,
    batch_size=1,
    variable_h=0.01,
)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
