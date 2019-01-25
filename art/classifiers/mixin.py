'''
Wrapper class for any classifier.
Subclass of the ClassifierMixin can override the behavior of
key functions, such as loss_gradient, to facilitate new attacks.
'''

class ClassifierMixin(object):
    """
    Wrapper class for any classifier instance
    """
    def __init__(self, classifier):
        """
        Initialize a `ClassifierMixin` object.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of an attack.
        """
        self.__classifier = classifier

    def __getattr__(self, attr):
        '''
        A generic grab-bag for the classifier instance
        This makes the wrapped class look like a subclass
        '''
        return getattr(self.__classifier, attr)

    def __setattr__(self, attr, value):
        '''
        A generic grab-bag for the classifier instance
        This makes the wrapped class look like a subclass
        '''
        if attr == '_ClassifierMixin__classifier':
            object.__setattr__(self, attr, value)
        else:
            setattr(self.__a, attr, value)
