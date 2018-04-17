"""
Classifier API for applying all attacks. Use the :class:`Classifier` wrapper to be able to apply an attack to a
preexisting model.
"""
from art.classifiers.classifier import Classifier
from art.classifiers.cnn import CNN
from art.classifiers.mlp import MLP
from art.classifiers.resnet import ResNet
