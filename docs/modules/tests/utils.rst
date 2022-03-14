:mod:`tests.utils`
===================
.. automodule:: tests.utils

Test Base Classes
-----------------
.. autoclass:: TestBase
.. autoclass:: ExpectedValue

Trained Models for Unittests, MNIST
-----------------------------------
.. autofunction:: get_image_classifier_tf
.. autofunction:: get_image_classifier_tf_v1
.. autofunction:: get_image_classifier_tf_v2
.. autofunction:: get_image_classifier_kr
.. autofunction:: get_image_classifier_kr_tf
.. autofunction:: get_image_classifier_kr_functional
.. autofunction:: get_image_classifier_kr_tf_functional
.. autofunction:: get_image_classifier_kr_tf_with_wildcard
.. autofunction:: get_image_classifier_kr_tf_binary
.. autofunction:: get_image_classifier_pt
.. autofunction:: get_classifier_bb
.. autofunction:: get_image_classifier_mxnet_custom_ini
.. autofunction:: get_gan_inverse_gan_ft

.. autofunction:: get_attack_classifier_pt

.. autofunction:: check_adverse_example_x
.. autofunction:: check_adverse_predicted_sample_y
.. autofunction:: is_valid_framework


Trained Models for Unittests, Iris
----------------------------------
.. autofunction:: get_tabular_classifier_tf
.. autofunction:: get_tabular_classifier_tf_v1
.. autofunction:: get_tabular_classifier_tf_v2
.. autofunction:: get_tabular_classifier_scikit_list
.. autofunction:: get_tabular_classifier_kr
.. autofunction:: get_tabular_classifier_pt

Random Number Generators
------------------------
.. autofunction:: master_seed
