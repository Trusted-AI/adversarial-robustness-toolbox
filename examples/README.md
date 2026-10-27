# Adversarial Robustness Toolbox examples

## Get Started with ART
These examples train a small model on the MNIST dataset and creates adversarial examples using the Fast Gradient Sign 
Method. Here we use the ART classifier to train the model, it would also be possible to provide a pretrained model to 
the ART classifier. The parameters are chosen for reduced computational requirements of the script and not optimised 
for accuracy.


### TensorFlow
[get_started_tensorflow_v2.py](get_started_tensorflow_v2.py) demonstrates a simple example of using ART with TensorFlow v2.x. 

### Keras
[get_started_keras.py](get_started_keras.py) demonstrates a simple example of using ART with Keras.

### PyTorch
[get_started_pytorch.py](get_started_pytorch.py) demonstrates a simple example of using ART with PyTorch.

### Scikit-learn
[get_started_scikit_learn.py](get_started_scikit_learn.py) demonstrates a simple example of using ART with Scikit-learn. 
This example uses the support vector machine SVC, but any other classifier of Scikit-learn can be used as well.

### XGBoost
[get_started_xgboost.py](get_started_xgboost.py) demonstrates a simple example of using ART with XGBoost.
Because gradient boosted tree classifier do not provide gradients, the adversarial examples are created with the 
black-box method Zeroth Order Optimization.

### InverseGAN
[get_started_inverse_gan.py](get_started_inverse_gan.py) demonstrates a simple example of using 
InverseGAN and Defense ART with TensorFlow v1.x.

### LightGBM
[get_started_lightgbm.py](get_started_lightgbm.py) demonstrates a simple example of using ART with LightGBM.
Because gradient boosted tree classifier do not provide gradients, the adversarial examples are created with the 
black-box method Zeroth Order Optimization.


## Applications

[adversarial_training_cifar10.py](adversarial_training_cifar10.py) trains a convolutional neural network on the CIFAR-10 
dataset, then generates adversarial images using the DeepFool attack and retrains the network on the training set 
augmented with the adversarial images.

[adversarial_training_data_augmentation.py](adversarial_training_data_augmentation.py) shows how to use ART and Keras 
to perform adversarial training using data generators for CIFAR-10.

[mnist_cnn_fgsm.py](mnist_cnn_fgsm.py) trains a convolutional neural network on MNIST, then crafts FGSM attack examples 
on it.

[mnist_poison_detection.py](mnist_poison_detection.py) generates a backdoor for MNIST dataset, then trains a 
convolutional neural network on the poisoned dataset and runs activation defence to find poison.
[fast_minimum_norm_digits.py](fast_minimum_norm_digits.py) evaluates L2 and Linf Fast Minimum-Norm (FMN) attacks on
100 held-out real handwritten digits, using a small trained classifier on CPU. The bundled scikit-learn dataset
requires no download. It saves metrics and exact samples; optionally compare with the authors' original FMN source
using `--reference-source /path/to/src/attacks/fmn.py` (requires Foolbox and EagerPy).
