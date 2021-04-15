# Adversarial Robustness Toolbox notebooks

## Expectation over Transformation (EoT)

[expectation_over_transformation_classification_rotation.ipynb](expectation_over_transformation_classification_rotation.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/expectation_over_transformation_classification_rotation.ipynb)]
show how to use Expectation over Transformation (EoT) sampling to make adversarial examples robust against rotation for image classification.


## Video Action Recognition

[adversarial_action_recognition.ipynb](adversarial_action_recognition.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_action_recognition.ipynb)]
shows how to create an adversarial attack on a video action recognition classification task with ART. Experiments in this notebook show how to modify a video sample by employing a Fast Gradient Method attack so that the modified video sample get mis-classified.

<p align="center">
  <img src="../utils/data/images/basketball.gif?raw=true" width="200" title="benign_basketball_sample">
  <img src="../utils/data/images/adversarial_basketball.gif?raw=true" width="200" title="adversarial_basketball_sample">
</p>


## Audio

[adversarial_audio_examples.ipynb](adversarial_audio_examples.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_audio_examples.ipynb)]
shows how to create adversarial examples of audio data with ART. Experiments in this notebook show how the waveform of a spoken digit of the AudioMNIST dataset can be modified with almost imperceptible changes so that the waveform gets mis-classified as different digit.

<p align="center">
  <img src="../utils/data/images/adversarial_audio_waveform.png?raw=true" width="200" title="adversarial_audio_waveform">
</p>

## Adversarial training

[adversarial_retraining.ipynb](adversarial_retraining.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_retraining.ipynb)]
shows how to load and evaluate the MNIST and CIFAR-10 models synthesized and adversarially trained by 
[Sinn et al., 2019](https://drive.google.com/uc?export=download&id=1XmUSqU7qCYigVqgEKvoT2p__Fy-Dq9Cx).

[adversarial_training_mnist.ipynb](adversarial_training_mnist.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_training_mnist.ipynb)]
demonstrates adversarial training of a neural network to harden the model against adversarial samples using the MNIST 
dataset.

<p align="center">
  <img src="../utils/data/images/adversarial_training.png?raw=true" width="200" title="adversarial_training">
</p>

## TensorFlow v2

[art-for-tensorflow-v2-callable.ipynb](art-for-tensorflow-v2-callable.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/art-for-tensorflow-v2-callable.ipynb)]
show how to use ART with TensorFlow v2 in eager execution mode with a model in form of a callable class or python 
function.

[art-for-tensorflow-v2-keras.ipynb](art-for-tensorflow-v2-keras.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/art-for-tensorflow-v2-keras.ipynb)]
demonstrates ART with TensorFlow v2 using tensorflow.keras without eager execution.

## Attacks

[attack_adversarial_patch.ipynb](adversarial_patch/attack_adversarial_patch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_adversarial_patch.ipynb)]
shows how to use ART to create real-world adversarial patches that fool real-world object detection and classification 
models.

<p align="center">
  <img src="../utils/data/images/adversarial_patch.png?raw=true" width="200" title="adversarial_patch">
</p>

[attack_decision_based_boundary.ipynb](attack_decision_based_boundary.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_decision_based_boundary.ipynb)]
demonstrates Decision-Based Adversarial Attack (Boundary) attack. This is a black-box attack which only requires class 
predictions.

[attack_decision_tree.ipynb](attack_decision_tree.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_decision_tree.ipynb)]
shows how to compute adversarial examples on decision trees ([Papernot et al., 2016](https://arxiv.org/abs/1605.07277)).
It traversing the structure of a decision tree classifier to create adversarial examples can be computed without 
explicit gradients.

[attack_defence_imagenet.ipynb](attack_defence_imagenet.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_defence_imagenet.ipynb)]
explains the basic workflow of using ART with defences and attacks on an neural network classifier for ImageNet.

[attack_hopskipjump.ipynb](attack_hopskipjump.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_hopskipjump.ipynb)]
demonstrates the HopSkipJumpAttack. This is a black-box attack that only requires class predictions. It is an advanced 
version of the Boundary attack.

## Classifiers

[classifier_blackbox.ipynb](classifier_blackbox.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_blackbox.ipynb)] demonstrates BlackBoxClassifier, the most general and
versatile classifier of ART requiring only a single predict function definition without any additional assumptions or 
requirements. The notebook shows how use BlackBoxClassifier to attack a remote, deployed model (in this case on IBM
Watson Machine Learning, https://cloud.ibm.com) using the HopSkiJump attack.

[classifier_blackbox_tesseract.ipynb](classifier_blackbox_tesseract.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_blackbox_tesseract.ipynb)]
demonstrates a black-box attack on Tesseract OCR. It uses BlackBoxClassifier and HopSkipJump attack to change the image 
of one word into the image of another word and shows how to apply pre-processing defences.

<p align="center">
  <img src="../utils/data/tesseract/assent_benign.png?raw=true" width="200" title="assent_benign">
  <img src="../utils/data/tesseract/assent_adversarial.png?raw=true" width="200" title="assent_adversarial">
</p>

[classifier_catboost.ipynb](classifier_catboost.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_catboost.ipynb)]
shows how to use ART with CatBoost models. It demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris
and MNIST datasets.

[classifier_gpy_gaussian_process.ipynb](classifier_gpy_gaussian_process.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_gpy_gaussian_process.ipynb)]
shows how to create adversarial examples for the Gaussian Process classifier of GPy. It crafts adversarial examples with
the HighConfidenceLowUncertainty (HCLU) attack ([Grosse et al., 2018](https://arxiv.org/abs/1812.02606)), specifically 
targeting Gaussian Process classifiers, and compares it to Projected Gradient Descent (PGD) 
([Madry et al., 2017](https://arxiv.org/abs/1706.06083)).

<p align="center">
  <img src="../utils/data/images/gaussian_process_hclu.png?raw=true" width="200" title="gaussian_process_hclu">
</p>

[classifier_lightgbm.ipynb](classifier_lightgbm.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_lightgbm.ipynb)]
shows how to use ART with LightGBM models. It demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris
and MNIST datasets.

[classifier_scikitlearn_AdaBoostClassifier.ipynb](classifier_scikitlearn_AdaBoostClassifier.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_AdaBoostClassifier.ipynb)]
shows how to use ART with Scikit-learn AdaBoostClassifier. It demonstrates and analyzes Zeroth Order Optimisation 
attacks using the Iris and MNIST datasets.

[classifier_scikitlearn_BaggingClassifier.ipynb](classifier_scikitlearn_BaggingClassifier.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_BaggingClassifier.ipynb)]
shows how to use ART with Scikit-learn BaggingClassifier. It demonstrates and analyzes Zeroth Order Optimisation attacks
using the Iris and MNIST datasets.

[classifier_scikitlearn_DecisionTreeClassifier.ipynb](classifier_scikitlearn_DecisionTreeClassifier.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_DecisionTreeClassifier.ipynb)]
shows how to use ART with Scikit-learn DecisionTreeClassifier. It demonstrates and analyzes Zeroth Order Optimisation 
attacks using the Iris and MNIST datasets.

[classifier_scikitlearn_ExtraTreesClassifier.ipynb](classifier_scikitlearn_ExtraTreesClassifier.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_ExtraTreesClassifier.ipynb)]
shows how to use ART with Scikit-learn ExtraTreesClassifier. It demonstrates and analyzes Zeroth Order Optimisation
attacks using the Iris and MNIST datasets.

[classifier_scikitlearn_GradientBoostingClassifier.ipynb](classifier_scikitlearn_GradientBoostingClassifier.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_GradientBoostingClassifier.ipynb)]
shows how to use ART with Scikit-learn GradientBoostingClassifier. It demonstrates and analyzes Zeroth Order 
Optimisation attacks using the Iris and MNIST datasets.

[classifier_scikitlearn_LogisticRegression.ipynb](classifier_scikitlearn_LogisticRegression.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_LogisticRegression.ipynb)]
shows how to use ART with Scikit-learn LogisticRegression. It demonstrates and analyzes Projected Gradient Descent 
attacks using the MNIST dataset.

[classifier_scikitlearn_pipeline_pca_cv_svc.ipynb](classifier_scikitlearn_pipeline_pca_cv_svc.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_pipeline_pca_cv_svc.ipynb)]
contains an example of generating adversarial examples using a black-box attack against a scikit-learn pipeline 
consisting of principal component analysis (PCA), cross validation (CV) and a support vector machine classifier (SVC), 
but any other valid pipeline would work too. The pipeline is optimised using grid search with cross validation. The 
adversarial samples are created with black-box HopSkipJump attack. The training data is MNIST, because of its intuitive 
visualisation, but any other dataset including tabular data would be suitable too.

[classifier_scikitlearn_RandomForestClassifier.ipynb](classifier_scikitlearn_RandomForestClassifier.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_RandomForestClassifier.ipynb)]
shows how to use ART with Scikit-learn RandomForestClassifier. It demonstrates and analyzes Zeroth Order Optimisation
attacks using the Iris and MNIST datasets.

[classifier_scikitlearn_SVC_LinearSVC.ipynb](classifier_scikitlearn_SVC_LinearSVC.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_scikitlearn_SVC_LinearSVC.ipynb)]
shows how to use ART with Scikit-learn SVC and LinearSVC support vector machines. It demonstrates and analyzes Projected 
Gradient Descent attacks using the Iris and MNIST dataset for binary and multiclass classification for linear and 
radial-basis-function kernels.

<p align="center">
  <img src="../utils/data/images/svc.png?raw=true" width="200" title="svc">
</p>

[classifier_xgboost.ipynb](classifier_xgboost.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_xgboost.ipynb)]
shows how to use ART with XGBoost models. It demonstrates and analyzes Zeroth Order Optimisation attacks using the Iris 
and MNIST datasets.

## Detectors

[detection_adversarial_samples_cifar10.ipynb](detection_adversarial_samples_cifar10.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/detection_adversarial_samples_cifar10.ipynb)]
demonstrates the detection of adversarial examples using ART. The classifier model is a neural network of a ResNet 
architecture in Keras for the CIFAR-10 dataset.

## Model stealing / model theft / model extraction

[model-stealing-demo.ipynb](model-stealing-demo.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/model-stealing-demo.ipynb)] demonstrates model stealing attacks and a reverse sigmoid defense against them.

## Poisoning

[poisoning_attack_svm.ipynb](poisoning_attack_svm.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_svm.ipynb)]
demonstrates a poisoning attack on a Support Vector Machine.

<p align="center">
  <img src="../utils/data/images/svm_poly.gif?raw=true" width="200" title="svm_poly">
</p>

[poisoning_defense_activation_clustering.ipynb](poisoning_defense_activation_clustering.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_activation_clustering.ipynb)]
demonstrates the generation and detection of backdoors in neural networks via Activation Clustering.

<p align="center">
  <img src="../utils/data/images/poisoning.png?raw=true" width="200" title="poisoning">
</p>

[poisoning_defense_neural_cleanse.ipynb](poisoning_defense_neural_cleanse.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_neural_cleanse.ipynb)]
demonstrates a defense against poisoning attacks that generation the suspected backdoor and applies runtime mitigation methods on the classifier.

[poisoning_defence_strip.ipynb](poisoning_defence_strip.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defence_strip.ipynb)]
demonstrates a defense against input-agnostic backdoor attacks that filters suspicious inputs at runtime.

## Certification and Verification

[output_randomized_smoothing_mnist.ipynb](output_randomized_smoothing_mnist.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/output_randomized_smoothing_mnist.ipynb)]
shows how to achieve certified adversarial robustness for neural networks via Randomized Smoothing.

[robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb](robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb)]
demonstrates the verification of adversarial robustness in decision tree ensemble classifiers 
(Gradient Boosted Decision Trees, Random Forests, etc.) using XGBoost, LightGBM and Scikit-learn.


## MNIST

[fabric_for_deep_learning_adversarial_samples_fashion_mnist.ipynb](fabric_for_deep_learning_adversarial_samples_fashion_mnist.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/fabric_for_deep_learning_adversarial_samples_fashion_mnist.ipynb)]
shows how to use ART with deep learning models trained with the Fabric for Deep Learning (FfDL).
