# ART Examples

[mnist_cnn_fgsm.py](mnist_cnn_fgsm.py)
Trains a convolutional neural network on MNIST, then crafts FGSM attack examples on it.

[mnist_transferability.py](mnist_transferability.py)
Trains a ResNet on the MNIST dataset, then generates adversarial images using DeepFool
and attacks a classic convolutional neural network with them.

[cifar_feature_squeezing.py](cifar_feature_squeezing.py)
Trains a convolutional neural network on the CIFAR10 dataset with feature squeezing as a defense.

[cifar_adversarial_training.py](cifar_adversarial_training.py)
Trains a convolutional neural network on the CIFAR-10 dataset, then generates adversarial images using the DeepFool attack and retrains the network on the training set augmented with the adversarial images.
