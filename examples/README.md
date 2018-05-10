# ART Examples

[mnist_cnn_fgsm.py](mnist_cnn_fgsm.py)
Trains a convolutional neural network on MNIST, then crafts FGSM attack examples on it.

[mnist_transferability.py](mnist_transferability.py)
Trains a convolutional neural network on the MNIST dataset using the Keras backend, then generates adversarial images using DeepFool
and uses them to attack a convolutional neural network trained on MNIST using TensorFlow. This is to show how to perform a
black-box attack: the attack never has access to the parameters of the TensorFlow model.

[cifar_adversarial_training.py](cifar_adversarial_training.py)
Trains a convolutional neural network on the CIFAR-10 dataset, then generates adversarial images using the
DeepFool attack and retrains the network on the training set augmented with the adversarial images.
