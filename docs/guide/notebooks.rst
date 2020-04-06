Notebooks
=========

Adversarial training
--------------------

`adversarial_retraining.ipynb`_ shows how to load and evaluate the MNIST and CIFAR-10 models synthesized and
adversarially trained by Sinn et al., 2019.

`adversarial_training_mnist.ipynb`_ demonstrates adversarial training of a neural network to harden the model against
adversarial samples using the MNIST dataset.

TensorFlow v2
-------------

`art-for-tensorflow-v2-callable.ipynb`_ show how to use ART with TensorFlow v2 in eager execution mode with a model in
form of a callable class or python function.

`art-for-tensorflow-v2-keras.ipynb`_ demonstrates ART with TensorFlow v2 using tensorflow.keras without eager execution.

Attacks
-------

`attack_adversarial_patch.ipynb`_ shows how to use ART to create real-world adversarial patches that fool real-world
object detection and classification models.

`attack_decision_based_boundary.ipynb`_ demonstrates Decision-Based Adversarial Attack (Boundary) attack. This is a
black-box attack which only requires class predictions.

`attack_decision_tree.ipynb`_ shows how to compute adversarial examples on decision trees (Papernot et al., 2016). It
traversing the structure of a decision tree classifier to create adversarial examples can be computed without explicit
gradients.

`attack_defence_imagenet.ipynb`_ explains the basic workflow of using ART with defences and attacks on an neural network
classifier for ImageNet.

`attack_hopskipjump.ipynb`_ demonstrates the HopSkipJumpAttack. This is a black-box attack that only requires class
predictions. It is an advanced version of the Boundary attack.

Classifiers
-----------

`classifier_blackbox.ipynb`_ demonstrates BlackBoxClassifier, the most general and versatile classifier of ART requiring
only a single predict function definition without any additional assumptions or requirements. The notebook shows how
use BlackBoxClassifier to attack a remote, deployed model (in this case on IBM Watson Machine Learning) using the
HopSkiJump attack.

`classifier_catboost.ipynb`_ shows how to use ART with CatBoost models. It demonstrates and analyzes Zeroth Order
Optimisation attacks using the Iris and MNIST datasets.

`classifier_gpy_gaussian_process.ipynb`_ shows how to create adversarial examples for the Gaussian Process classifier of
GPy. It crafts adversarial examples with the HighConfidenceLowUncertainty (HCLU) attack (Grosse et al., 2018),
specifically targeting Gaussian Process classifiers, and compares it to Projected Gradient Descent (PGD)
(Madry et al., 2017).

`classifier_lightgbm.ipynb`_ shows how to use ART with LightGBM models. It demonstrates and analyzes Zeroth Order
Optimisation attacks using the Iris and MNIST datasets.

`classifier_scikitlearn_AdaBoostClassifier.ipynb`_ shows how to use ART with Scikit-learn AdaBoostClassifier. It
demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris and MNIST datasets.

`classifier_scikitlearn_BaggingClassifier.ipynb`_ shows how to use ART with Scikit-learn BaggingClassifier. It
demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris and MNIST datasets.

`classifier_scikitlearn_DecisionTreeClassifier.ipynb`_ shows how to use ART with Scikit-learn DecisionTreeClassifier.
It demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris and MNIST datasets.

`classifier_scikitlearn_ExtraTreesClassifier.ipynb`_ shows how to use ART with Scikit-learn ExtraTreesClassifier. It
demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris and MNIST datasets.

`classifier_scikitlearn_GradientBoostingClassifier.ipynb`_ shows how to use ART with Scikit-learn
GradientBoostingClassifier. It demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris and MNIST
datasets.

`classifier_scikitlearn_LogisticRegression.ipynb`_ shows how to use ART with Scikit-learn LogisticRegression. It
demonstrates and analyzes Projected Gradient Descent attacks using the MNIST dataset.

`classifier_scikitlearn_pipeline_pca_cv_svc.ipynb`_ contains an example
of generating adversarial examples using a black-box attack against a scikit-learn pipeline consisting of principal
component analysis (PCA), cross validation (CV) and a support vector machine classifier (SVC), but any other valid
pipeline would work too. The pipeline is optimised using grid search with cross validation. The adversarial samples are
created with black-box HopSkipJump attack. The training data is MNIST, because of its intuitive visualisation, but any
other dataset including tabular data would be suitable too.

`classifier_scikitlearn_RandomForestClassifier.ipynb`_ shows
how to use ART with Scikit-learn RandomForestClassifier. It demonstrates and analyzes Zeroth Order Optimisation attacks
using the Iris and MNIST datasets.

`classifier_scikitlearn_SVC_LinearSVC.ipynb`_ shows
how to use ART with Scikit-learn SVC and LinearSVC support vector machines. It demonstrates and analyzes Projected
Gradient Descent attacks using the Iris and MNIST dataset for binary and multiclass classification for linear and
radial-basis-function kernels.

`classifier_xgboost.ipynb`_ shows how to use ART with XGBoost models. It demonstrates and analyzes Zeroth Order
Optimisation attacks using the Iris and MNIST datasets.

Detectors
---------

`detection_adversarial_samples_cifar10.ipynb`_ demonstrates the detection of
adversarial examples using ART. The classifier model is a neural network of a ResNet architecture in Keras for the
CIFAR-10 dataset.

Poisoning
---------

`poisoning_dataset_mnist.ipynb`_ demonstrates the generation and detection of backdoors attacks into neural networks by
poisoning the training dataset.

`poisoning_attack_svm.ipynb`_ demonstrates the generation of malicious poisoning examples on Support Vector Machines.

`poisoning_attack_feature_collision.ipynb`_ demonstrates the generation of a feature collision clean-label attack on a
Keras classifier.

Certification and Verification
------------------------------

`output_randomized_smoothing_mnist.ipynb`_ shows how to achieve certified
adversarial robustness for neural networks via Randomized Smoothing.

`robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb`_
demonstrates the verification of adversarial robustness in decision tree ensemble classifiers (Gradient Boosted Decision
Trees, Random Forests, etc.) using XGBoost, LightGBM and Scikit-learn.


.. _adversarial_retraining.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/adversarial_retraining.ipynb
.. _adversarial_training_mnist.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/adversarial_training_mnist.ipynb
.. _art-for-tensorflow-v2-callable.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/art-for-tensorflow-v2-callable.ipynb
.. _art-for-tensorflow-v2-keras.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/art-for-tensorflow-v2-keras.ipynb
.. _attack_adversarial_patch.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/attack_adversarial_patch.ipynb
.. _attack_decision_based_boundary.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/attack_decision_based_boundary.ipynb
.. _attack_decision_tree.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/attack_decision_tree.ipynb
.. _attack_defence_imagenet.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/attack_defence_imagenet.ipynb
.. _attack_hopskipjump.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/attack_hopskipjump.ipynb
.. _classifier_blackbox.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_blackbox.ipynb
.. _classifier_catboost.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_catboost.ipynb
.. _classifier_gpy_gaussian_process.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_gpy_gaussian_process.ipynb
.. _classifier_lightgbm.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_lightgbm.ipynb
.. _classifier_scikitlearn_AdaBoostClassifier.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_AdaBoostClassifier.ipynb
.. _classifier_scikitlearn_BaggingClassifier.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_BaggingClassifier.ipynb
.. _classifier_scikitlearn_DecisionTreeClassifier.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_DecisionTreeClassifier.ipynb
.. _classifier_scikitlearn_ExtraTreesClassifier.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_ExtraTreesClassifier.ipynb
.. _classifier_scikitlearn_GradientBoostingClassifier.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_GradientBoostingClassifier.ipynb
.. _classifier_scikitlearn_LogisticRegression.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_LogisticRegression.ipynb
.. _classifier_scikitlearn_pipeline_pca_cv_svc.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_pipeline_pca_cv_svc.ipynb
.. _classifier_scikitlearn_RandomForestClassifier.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_RandomForestClassifier.ipynb
.. _classifier_scikitlearn_SVC_LinearSVC.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_scikitlearn_SVC_LinearSVC.ipynb
.. _classifier_xgboost.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/classifier_xgboost.ipynb
.. _detection_adversarial_samples_cifar10.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/detection_adversarial_samples_cifar10.ipynb
.. _poisoning_dataset_mnist.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/poisoning_dataset_mnist.ipynb
.. _poisoning_attack_svm.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/poisoning_attack_svm.ipynb
.. _poisoning_attack_feature_collision.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/poisoning_attack_feature_collision.ipynb
.. _output_randomized_smoothing_mnist.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/output_randomized_smoothing_mnist.ipynb
.. _robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb: https://github.com/IBM/adversarial-robustness-toolbox/blob/master/notebooks/robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb
