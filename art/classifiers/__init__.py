"""
Classifier API for applying all attacks. Use the :class:`Classifier` wrapper to be able to apply an attack to a
preexisting model.
"""
from art.classifiers.classifier import Classifier, ImageClassifier, TextClassifier
from art.classifiers.keras import KerasClassifier, KerasImageClassifier, KerasTextClassifier
from art.classifiers.mxnet import MXClassifier, MXImageClassifier, MXTextClassifier
from art.classifiers.tensorflow import TFClassifier, TFImageClassifier, TFTextClassifier

# PyTorchClassifier class creation requires `torch` install; create class only if `torch` is present
# Otherwise, silence `torch` import error until the user actually tries to use the class
try:
    from art.classifiers.pytorch import PyTorchClassifier, PyTorchImageClassifier, PyTorchTextClassifier
except ImportError:
    pass
