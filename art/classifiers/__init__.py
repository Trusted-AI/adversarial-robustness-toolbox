"""
Classifier API for applying all attacks. Use the :class:`Classifier` wrapper to be able to apply an attack to a
preexisting model.
"""
from art.classifiers.classifier import Classifier
from art.classifiers.keras import KerasClassifier
from art.classifiers.mxnet import MXClassifier
from art.classifiers.tensorflow import TFClassifier

# PyTorchClassifier class creation requires `torch` install; create class only if `torch` is present
# Otherwise, silence `torch` import error until the user actually tries to use the class
try:
    from art.classifiers.pytorch import PyTorchClassifier
except ImportError:
    pass
