import numpy as np

def label_smoothing(y_val, max_value=0.9):

    min_value = (1 - max_value) / (y_val.shape[1] - 1)

    assert max_value >= min_value

    smoothed_y = y_val.copy()
    smoothed_y[smoothed_y == 1.] = max_value
    smoothed_y[smoothed_y == 0.] = min_value

    return smoothed_y

def feature_squeezing(x_val, bit_depth=8):
    """ x_val is supposed to be in the range [0,1] """

    assert 60 > bit_depth > 0

    max_value = int(2**bit_depth - 1)

    squeezed_x = np.rint(x_val*max_value) / max_value

    return squeezed_x