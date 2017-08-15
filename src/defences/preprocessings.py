import numpy as np
import tensorflow as tf

def label_smoothing(y_val, max_value=0.9):
    """
    Computes a vector of smooth labels from a vector of hard ones. 
    :param y_val: (np.ndarray) original vector of label probabilities
    :param max_value: (float) probability to affect to label of probability 1
    :return: (np.ndarray) vector of smooth probabilities
    """

    min_value = (1 - max_value) / (y_val.shape[1] - 1)

    assert max_value >= min_value

    smoothed_y = y_val.copy()
    smoothed_y[smoothed_y == 1.] = max_value
    smoothed_y[smoothed_y == 0.] = min_value

    return smoothed_y

def feature_squeezing(x_val, bit_depth=8):
    """
    Reduces the sensibility of the features of a sample.
    Defence method from https://arxiv.org/abs/1704.01155.    
    :param x_val: (np.ndarray) Sample to squeeze. `x_val` values are supposed to be in the range [0,1]
    :param bit_depth: (int) sensibility magnitude
    :return: (np.ndarray) squeezed sample
    """

    assert 60 > bit_depth > 0

    max_value = int(2 ** bit_depth - 1)

    squeezed_x = np.rint(x_val*max_value) / max_value

    return squeezed_x

def tf_feature_squeezing(x, bit_depth=8):
    """ 
    feature squeezing on tf.Tensor. See `src.defences.preprocessings.feature_squeezing` for documentation.
    """

    assert 60 > bit_depth > 0

    max_value = int(2 ** bit_depth - 1)

    x = tf.rint(x * max_value) / max_value

    return x