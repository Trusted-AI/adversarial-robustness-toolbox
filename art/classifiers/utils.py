from __future__ import absolute_import, division, print_function, unicode_literals


def check_is_fitted(model, parameters, error_msg=None):
    """
    Checks if the model is fitted by asserting the presence of the fitted parameters in the model.

    :param model: The model instance
    :param parameters: The name of the parameter or list of names of parameters that are fitted by the model
    :param error_msg: (string) Custom error message to be printed if the model is not fitted. Default message is
    'This model is not fitted yet. Call 'fit' with appropriate arguments before using this method.'
    :return: (bool) True if the model is fitted
    :raises: TypeError
    """
    if error_msg is None:
        error_msg = "This model is not fitted yet. Call 'fit' with appropriate arguments before using this method."

    if not hasattr(model, 'fit'):
        raise TypeError("%s cannot be fitted." % model)

    if not isinstance(parameters, (list, tuple)):
        parameters = [parameters]

    if not all([hasattr(model, param) for param in parameters]):
        return False

    return True
