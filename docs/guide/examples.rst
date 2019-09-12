Examples
========

The examples of ART are located in directory `examples` in the ART `GitHub`_ repository. Each example can be run with the
following command:

.. code-block:: bash

   python examples/<example_name>.py


Get Started with ART
--------------------
These examples train a small model on the MNIST dataset and creates adversarial examples using the Fast Gradient Sign
Method. Here we use the ART classifier to train the model, it would also be possible to provide a pretrained model to
the ART classifier. The parameters are chosen for reduced computational requirements of the script and not optimised
for accuracy.


TensorFlow
^^^^^^^^^^
`get_started_tensorflow.py`_ demonstrates a simple example of using ART with TensorFlow v1.x.

Keras
^^^^^
`get_started_keras.py`_ demonstrates a simple example of using ART with Keras.

PyTorch
^^^^^^^
`get_started_pytorch.py`_ demonstrates a simple example of using ART with PyTorch.

MXNet
^^^^^
`get_started_mxnet.py`_ demonstrates a simple example of using ART with MXNet.

Scikit-learn
^^^^^^^^^^^^
`get_started_scikit_learn.py`_ demonstrates a simple example of using ART with Scikit-learn.
This example uses the support vector machine SVC, but any other classifier of Scikit-learn can be used as well.

XGBoost
^^^^^^^
`get_started_xgboost.py`_ demonstrates a simple example of using ART with XGBoost.
Because gradient boosted tree classifier do not provide gradients, the adversarial examples are created with the
black-box method Zeroth Order Optimization.

LightGBM
^^^^^^^^
`get_started_lightgbm.py`_ demonstrates a simple example of using ART with LightGBM.
Because gradient boosted tree classifier do not provide gradients, the adversarial examples are created with the
black-box method Zeroth Order Optimization.


Applications
------------

`adversarial_training_cifar10.py`_ trains a convolutional neural network on the CIFAR-10
dataset, then generates adversarial images using the DeepFool attack and retrains the network on the training set
augmented with the adversarial images.

`adversarial_training_data_augmentation.py`_ shows how to use ART and Keras to perform adversarial
training using data generators for CIFAR-10.

`mnist_cnn_fgsm.py`_ trains a convolutional neural network on MNIST, then crafts FGSM attack examples on it.

`mnist_poison_detection.py`_ generates a backdoor for MNIST dataset, then trains a convolutional neural network on the
poisoned dataset and runs activation defence to find poison.

`mnist_transferability.py`_ trains a convolutional neural network on the MNIST dataset using the Keras backend, then
generates adversarial images using DeepFool and uses them to attack a convolutional neural network trained on MNIST
using TensorFlow. This is to show how to perform a black-box attack: the attack never has access to the parameters of
the TensorFlow model.


.. _GitHub: https://github.com/IBM/adversarial-robustness-toolbox
.. _get_started_tensorflow.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/get_started_tensorflow.py
.. _get_started_keras.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/get_started_keras.py
.. _get_started_pytorch.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/get_started_pytorch.py
.. _get_started_mxnet.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/get_started_mxnet.py
.. _get_started_scikit_learn.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/get_started_scikit_learn.py
.. _get_started_xgboost.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/get_started_xgboost.py
.. _get_started_lightgbm.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/get_started_lightgbm.py
.. _adversarial_training_cifar10.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/adversarial_training_cifar10.py
.. _adversarial_training_data_augmentation.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/adversarial_training_data_augmentation.py
.. _mnist_cnn_fgsm.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/mnist_cnn_fgsm.py
.. _mnist_poison_detection.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/mnist_poison_detection.py
.. _mnist_transferability.py: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/examples/mnist_transferability.py
