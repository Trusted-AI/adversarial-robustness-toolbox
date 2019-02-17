"""
Wrapper class for any classifier.
Subclass of the ClassifierWrapper can override the behavior of
key functions, such as loss_gradient, to facilitate new attacks.
"""

class ClassifierWrapper(object):
    """
    Wrapper class for any classifier instance
    """
    attack_params = ['__classifier']

    def __init__(self, classifier):
        """
        Initialize a `ClassifierWrapper` object.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of an attack.
        """
        self.__classifier = classifier

    def __getattr__(self, attr):
        """
        A generic grab-bag for the classifier instance
        This makes the wrapped class look like a subclass
        """
        return getattr(self.__classifier, attr)

    def __setattr__(self, attr, value):
        """
        A generic grab-bag for the classifier instance
        This makes the wrapped class look like a subclass
        """
        if attr == '_ClassifierWrapper__classifier':
            object.__setattr__(self, attr, value)
        else:
            setattr(self.__classifier, attr, value)
    
    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True
