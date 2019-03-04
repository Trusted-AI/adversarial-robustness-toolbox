"""
Wrapper class for any classifier.
Subclass of the ClassifierWrapper can override the behavior of
key functions, such as loss_gradient, to facilitate new attacks.
"""

class ClassifierWrapper():
    """
    Wrapper class for any classifier instance
    """
    attack_params = ['_classifier']

    def __init__(self, classifier):
        """
        Initialize a `ClassifierWrapper` object.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of an attack.
        """
        self._classifier = classifier

    def __getattr__(self, attr):
        """
        A generic grab-bag for the classifier instance
        This makes the wrapped class look like a subclass
        """
        return getattr(self._classifier, attr)

    def __setattr__(self, attr, value):
        """
        A generic grab-bag for the classifier instance
        This makes the wrapped class look like a subclass
        """
        if attr == '_classifier':
            object.__setattr__(self, attr, value)
        else:
            setattr(self._classifier, attr, value)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and pass them down to the underlying
        wrapped classifier instance.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True
