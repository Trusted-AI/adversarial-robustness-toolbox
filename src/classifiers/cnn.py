from cleverhans.utils import conv_2d

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

def cnn_model(input_shape, logits=False, input_ph=None, nb_filters=64, nb_classes=10):
    """
    Returns a CNN model using Keras sequential model
    
    # Arguments
        logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
        input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
        img_rows: number of row in the image
        img_cols: number of columns in the image
        channels: number of color channels (e.g., 1 for MNIST)
        nb_filters: number of convolutional filters per layer
        nb_classes: the number of output classes
    # Return
        CNN model
    """

    model = Sequential()

    layers = [conv_2d(nb_filters, (8, 8), (2, 2), "same", input_shape=input_shape),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model