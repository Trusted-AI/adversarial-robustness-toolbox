def label_smoothing(y_val, max_value=0.9):

    min_value = (1 - max_value) / (y_val.shape[1] - 1)

    assert max_value >= min_value

    smoothed_y = y_val.copy()
    smoothed_y[smoothed_y == 1.] = max_value
    smoothed_y[smoothed_y == 0.] = min_value

    return smoothed_y