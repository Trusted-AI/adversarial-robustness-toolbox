"""
Classifier API for applying all attacks. Use the :class:`Classifier` wrapper to be able to apply an attack to a
preexisting model.
"""
from src.classifiers.classifier import Classifier
from src.classifiers.keras import KerasClassifier
from src.classifiers.tensorflow import TFClassifier
