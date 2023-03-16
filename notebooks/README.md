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

[poisoning_attack_backdoor_audio.ipynb](poisoning_attack_backdoor_audio.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_backdoor_audio.ipynb)]
demonstrates the dirty-label backdoor attack on a TensorflowV2 estimator for speech classification.

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
[attack_feature_adversaries_pytorch.ipynb](attack_feature_adversaries_pytorch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_feature_adversaries_pytorch.ipynb)]
or [attack_feature_adversaries_tensorflow_v2.ipynb](attack_feature_adversaries_tensorflow_v2.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_feature_adversaries_tensorflow_v2.ipynb)]
show how to use ART to create feature adversaries ([Sabour et al., 2016](https://arxiv.org/abs/1511.05122)).

[attack_adversarial_patch.ipynb](adversarial_patch/attack_adversarial_patch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_patch/attack_adversarial_patch.ipynb)]
shows how to use ART to create real-world adversarial patches that fool real-world object detection and classification 
models.
[attack_adversarial_patch_TensorFlowV2.ipynb](adversarial_patch/attack_adversarial_patch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_patch/attack_adversarial_patch_TensorFlowV2.ipynb)]  TensorFlow v2 specific attack implementation. 
[attack_adversarial_patch_pytorch_yolo.ipynb](adversarial_patch/attack_adversarial_patch_pytorch_yolo.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_patch/attack_adversarial_patch_pytorch_yolo.ipynb)] YOLO v3 and v5 specific attack.

[attack_adversarial_patch_faster_rcnn.ipynb](adversarial_patch/attack_adversarial_patch_faster_rcnn.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_patch/attack_adversarial_patch_faster_rcnn.ipynb)]
shows how to set up a TFv2 Faster R-CNN object detector with ART and create an adversarial patch attack that fools the detector.

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

[attack_membership_inference.ipynb](attack_membership_inference.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference.ipynb)]
demonstrates the MembershipInferenceBlackBoxRuleBased and MembershipInferenceBlackBox membership inference attacks on a classifier model with only black-box access.

[attack_membership_inference_regressor.ipynb](attack_membership_inference_regressor.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference_regressor.ipynb)]
demonstrates the MembershipInferenceBlackBox membership inference attack on a regressor model with only black-box access.

[attack_attribute_inference.ipynb](attack_attribute_inference.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_attribute_inference.ipynb)]
demonstrates the AttributeInferenceBlackBox, AttributeInferenceWhiteBoxLifestyleDecisionTree and AttributeInferenceWhiteBoxDecisionTree attribute inference attacks on a classifier model.

[attack_attribute_inference_regressor.ipynb](attack_attribute_inference_regressor.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_attribute_inference_regressor.ipynb)]
demonstrates the AttributeInferenceBlackBox attribute inference attacks on a regressor model.

[attack_membership_inference_shadow_models.ipynb](attack_membership_inference_shadow_models.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference_shadow_models.ipynb)]
demonstrates a MembershipInferenceBlackBox membership inference attack using shadow models on a classifier.

[label_only_membership_inference.ipynb](label_only_membership_inference.ipynb) [[on nbviewer](https://nbviewer.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/label_only_membership_inference.ipynb)]
demonstrates a LabelOnlyDecisionBoundary membership inference attack on a PyTorch classifier for the MNIST dataset.

## Metrics

[privacy_metric.ipynb](privacy_metric.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/privacy_metric.ipynb)] 
demonstrates how to apply both the PDTP and the SHAPr privacy metrics to random forest and decision tree classifiers
trained on the nursery dataset.

## Classifiers

[classifier_blackbox.ipynb](classifier_blackbox.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_blackbox.ipynb)] demonstrates BlackBoxClassifier, the most general and
versatile classifier of ART requiring only a single predict function definition without any additional assumptions or 
requirements. The notebook shows how use BlackBoxClassifier to attack a remote, deployed model (in this case on IBM
Watson Machine Learning, https://cloud.ibm.com) using the HopSkiJump attack.

[classifier_blackbox_lookup_table.ipynb](classifier_blackbox_lookup_table.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/classifier_blackbox_lookup_table.ipynb)]
demonstrates using BlackBoxClassifier when the adversary does not have access to the model for making predictions, but
does have a set of existing predictions produced before losing access. The notebook shows how to use BlackBoxClassifier
to attack a model using only a table of samples and their labels, using a membership inference black-box attack.

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

[hidden_trigger_backdoor/poisoning_attack_hidden_trigger_pytorch.ipynb](poisoning_attack_hidden_trigger_pytorch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/hidden_trigger_backdoor/poisoning_attack_hidden_trigger_pytorch.ipynb)]
demonstrates the Hidden Trigger Backdoor attack on a PyTorch estimator.

[hidden_trigger_backdoor/poisoning_attack_hidden_trigger_keras.ipynb](poisoning_attack_hidden_trigger_keras.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/hidden_trigger_backdoor/poisoning_attack_hidden_trigger_keras.ipynb)]
demonstrates the Hidden Trigger Backdoor attack on a Keras estimator.

[hidden_trigger_backdoor/poisoning_attack_hidden_trigger_tf.ipynb](poisoning_attack_hidden_trigger_pytorch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/hidden_trigger_backdoor/poisoning_attack_hidden_trigger_tf.ipynb)]
demonstrates the Hidden Trigger Backdoor attack on a TensorflowV2 estimator.

[poisoning_defense_activation_clustering.ipynb](poisoning_defense_activation_clustering.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_activation_clustering.ipynb)]
demonstrates the generation and detection of backdoors in neural networks via Activation Clustering.

<p align="center">
  <img src="../utils/data/images/poisoning.png?raw=true" width="200" title="poisoning">
</p>

[poisoning_defense_deep_partition_aggregation.ipynb](poisoning_defense_deep_partition_aggregation.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_deep_partition_aggregation.ipynb)]
demonstrates a defense against poisoning attacks via partitioning the data into disjoint subsets and training an ensemble model.

[poisoning_defense_dp_instahide.ipynb](poisoning_defense_dp_instahide.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_dp_instahide.ipynb)]
demonstrates a defense against poisoning attacks using the DP-InstaHide training method which uses data augmentation and additive noise.

[poisoning_defense_neural_cleanse.ipynb](poisoning_defense_neural_cleanse.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_neural_cleanse.ipynb)]
demonstrates a defense against poisoning attacks that generation the suspected backdoor and applies runtime mitigation methods on the classifier.

[poisoning_defence_strip.ipynb](poisoning_defence_strip.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defence_strip.ipynb)]
demonstrates a defense against input-agnostic backdoor attacks that filters suspicious inputs at runtime.

[poisoning_attack_witches_brew.ipynb](poisoning_attack_witches_brew.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_witches_brew.ipynb)]
demonstrates the gradient matching poisoning attack (a.k.a. Witches' Brew) that adds noise to align the training gradient to a specific direction that can poison the target model.

[poisoning_attack_feature_collision.ipynb](poisoning_attack_feature_collision.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_feature_collision.ipynb)]
demonstrates working Poison Frog (Feature Collision) poisoning attack implemented in Keras Framework on CIFAR10 dataset as per the ([paper](https://arxiv.org/pdf/1804.00792.pdf)). This is a targeted clean label attack, which do not require the attacker to have any control over the labeling of training data and control the behavior of the classifier on a specific test instance without degrading overall classifier performance.

[poisoning_attack_feature_collision-pytorch.ipynb](poisoning_attack_feature_collision-pytorch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_feature_collision-pytorch.ipynb)]
demonstrates working Poison Frog (Feature Collision) poisoning attack implemented in PyTorch Framework on CIFAR10 dataset as per the ([paper](https://arxiv.org/pdf/1804.00792.pdf)). This is a targeted clean label attack, which do not require the attacker to have any control over the labeling of training data and control the behavior of the classifier on a specific test instance without degrading overall classifier performance.

[poisoning_attack_sleeper_agent_pytorch.ipynb](poisoning_attack_sleeper_agent_pytorch.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_sleeper_agent_pytorch.ipynb)]
demonstrates working Sleeper Agent poisoning attack implemented in PyTorch Framework on CIFAR10 dataset as per the ([paper](https://arxiv.org/pdf/2106.08970.pdf)). A new hidden trigger attack, Sleeper Agent,
which employs gradient matching, data selection, and target model re-training during the crafting process. Sleeper
Agent is the first hidden trigger backdoor attack to be effective against neural networks trained from scratch.

[poisoning_attack_bad_det.ipynb](poisoning_attack_bad_det.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_attack_bad_det.ipynb)]
demonstrates using the BadDet poisoning attacks to insert backdoors and create poisoned samples for object detector models. This is a dirty label attack where a trigger is inserted into a bounding box and the classification labels are changed accordingly.

## Certification and Verification

[output_randomized_smoothing_mnist.ipynb](output_randomized_smoothing_mnist.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/output_randomized_smoothing_mnist.ipynb)]
shows how to achieve certified adversarial robustness for neural networks via Randomized Smoothing.

[robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb](robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/robustness_verification_clique_method_tree_ensembles_gradient_boosted_decision_trees_classifiers.ipynb)]
demonstrates the verification of adversarial robustness in decision tree ensemble classifiers 
(Gradient Boosted Decision Trees, Random Forests, etc.) using XGBoost, LightGBM and Scikit-learn.

[certification_deepz.ipynb](certification_deepz.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/certification_deepz.ipynb)]
demonstrates using DeepZ to compute certified robustness for neural networks.

<p align="center">
  <img src="../utils/data/images/zonotope_picture.png?raw=true" width="200" title="deepz">
</p>

## Certified Training

[certified_adversarial_training.ipynb](certified_adversarial_training.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/certified_adversarial_training.ipynb)]
Demonstrates training a neural network for certified robustness using bound propagation techniques.

<p align="center">
  <img src="../utils/data/images/cert_training.png?raw=true" width="200" title="certified training">
</p>

[certification_interval_domain.ipynb](certification_interval_domain.ipynb)[[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/certification_interval_domain.ipynb)]
demonstrates using interval bound propagation for certification of neural network robustness.
<p align="center">
  <img src="../utils/data/images/IBP_certification.png?raw=true" width="200" title="IBP certification">
</p>

## MNIST

[fabric_for_deep_learning_adversarial_samples_fashion_mnist.ipynb](fabric_for_deep_learning_adversarial_samples_fashion_mnist.ipynb) [[on nbviewer](https://nbviewer.jupyter.org/github/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/fabric_for_deep_learning_adversarial_samples_fashion_mnist.ipynb)]
shows how to use ART with deep learning models trained with the Fabric for Deep Learning (FfDL).
