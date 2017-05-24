import os

from config import DATA_PATH
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

from src.classifiers import cnn
from src.utils import get_args, get_verbose_print, load_mnist, make_directory

# --------------------------------------------------------------------------------------------------- SETTINGS
args = get_args(__file__)

v_print = get_verbose_print(args.verbose)

comp_params = {"loss": 'categorical_crossentropy',
               "optimizer": 'adam',
               "metrics": ['accuracy']}

# --------------------------------------------------------------------------------------------- GET CLASSIFIER
# get MNIST
(X_train, Y_train), (X_test, Y_test) = load_mnist()
# X_train, Y_train, X_test, Y_test = X_train[:1000], Y_train[:1000], X_test[:1000], Y_test[:1000]
im_shape = X_train[0].shape

session = tf.Session()
K.set_session(session)

if args.load:
    MODEL_PATH = os.path.join(os.path.abspath(args.load), "")

    model = cnn.load_model(MODEL_PATH, "best-weights.h5")

else:
    MODEL_PATH = os.path.join(os.path.abspath(DATA_PATH), "classifiers", "mnist", "cnn", args.act, "")
    model = cnn.cnn_model(im_shape, act=args.act, bnorm=False)

    # Fit the model

    model.compile(**comp_params)

    if args.save:

        make_directory(MODEL_PATH)

        # Save best model weights
        # checkpoint = ModelCheckpoint(os.path.join(FILEPATH,"best-weights.{epoch:02d}-{val_acc:.2f}.h5"),
        #                              monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, "best-weights.h5"), monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')

        # Remote monitor
        monitor = TensorBoard(log_dir=os.path.join(MODEL_PATH, 'logs'), write_graph=False)

        callbacks_list = [checkpoint, monitor]
    else:
        callbacks_list = []

    model.fit(X_train, Y_train, verbose=2*int(args.verbose), validation_split=args.val_split, epochs=args.nb_epochs,
              batch_size=args.batch_size, callbacks=callbacks_list)

    if args.save:
        cnn.save_model(model, MODEL_PATH, comp_params)
        # Load model with best validation score
        model = cnn.load_model(MODEL_PATH, "best-weights.h5")

scores = model.evaluate(X_test, Y_test, verbose=args.verbose)
v_print("\naccuracy: %.2f%%" % (scores[1] * 100))
