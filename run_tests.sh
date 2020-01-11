#!/usr/bin/env bash
exit_code=0

# attacks

echo "######################"
echo "test_adversarial_patch"
echo "######################"
coverage run 'tests/attacks/test_adversarial_patch.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_adversarial_patch.py'; fi

echo "#############"
echo "test_boundary"
echo "#############"
coverage run 'tests/attacks/test_boundary.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_boundary.py'; fi

echo "############"
echo "test_carlini"
echo "############"
coverage run 'tests/attacks/test_carlini.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_carlini.py'; fi

echo "#########################"
echo "test_decision_tree_attack"
echo "#########################"
coverage run 'tests/attacks/test_decision_tree_attack.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_decision_tree_attack.py'; fi

echo "#############"
echo "test_deepfool"
echo "#############"
coverage run 'tests/attacks/test_deepfool.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_deepfool.py'; fi

echo "################"
echo "test_elastic_net"
echo "################"
coverage run 'tests/attacks/test_elastic_net.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_elastic_net.py'; fi

echo "##################"
echo "test_fast_gradient"
echo "##################"
coverage run 'tests/attacks/test_fast_gradient.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_fast_gradient.py'; fi

echo "#######################################"
echo "test_functionally_equivalent_extraction"
echo "#######################################"
coverage run 'tests/attacks/test_functionally_equivalent_extraction.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_functionally_equivalent_extraction.py'; fi

echo "#########"
echo "test_hclu"
echo "#########"
coverage run 'tests/attacks/test_hclu.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_hclu.py'; fi

echo "##################"
echo "test_hop_skip_jump"
echo "##################"
coverage run 'tests/attacks/test_hop_skip_jump.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_hop_skip_jump.py'; fi

echo "#####################"
echo "test_iterative_method"
echo "#####################"
coverage run 'tests/attacks/test_iterative_method.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_iterative_method.py'; fi

echo "###############"
echo "test_newtonfool"
echo "###############"
coverage run 'tests/attacks/test_newtonfool.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_newtonfool.py'; fi

echo "#########################"
echo "test_poisoning_attack_svm"
echo "#########################"
coverage run 'tests/attacks/test_poisoning_attack_svm.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_poisoning_attack_svm.py'; fi

echo "###############################"
echo "test_projected_gradient_descent"
echo "###############################"
coverage run 'tests/attacks/test_projected_gradient_descent.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_projected_gradient_descent.py'; fi

echo "#################"
echo "test_saliency_map"
echo "#################"
coverage run 'tests/attacks/test_saliency_map.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_saliency_map.py'; fi

echo "###########################"
echo "test_spatial_transformation"
echo "###########################"
coverage run 'tests/attacks/test_spatial_transformation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_spatial_transformation.py'; fi

echo "###########################"
echo "test_universal_perturbation"
echo "###########################"
coverage run 'tests/attacks/test_universal_perturbation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_universal_perturbation.py'; fi

echo "########################"
echo "test_virtual_adversarial"
echo "########################"
coverage run 'tests/attacks/test_virtual_adversarial.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_virtual_adversarial.py'; fi

echo "########"
echo "test_zoo"
echo "########"
coverage run 'tests/attacks/test_zoo.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_zoo.py'; fi

# classifiers
echo "#############"
echo "test_blackbox"
echo "#############"
coverage run 'tests/classifiers/test_blackbox.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_blackbox.py'; fi

echo "#############"
echo "test_catboost"
echo "#############"
coverage run 'tests/classifiers/test_catboost.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_catboost.py'; fi

echo "###############"
echo "test_classifier"
echo "###############"
coverage run 'tests/classifiers/test_classifier.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_classifier.py'; fi

echo "########################"
echo "test_detector_classifier"
echo "########################"
coverage run 'tests/classifiers/test_detector_classifier.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_detector_classifier.py'; fi

echo "#############"
echo "test_ensemble"
echo "#############"
coverage run 'tests/classifiers/test_ensemble.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_ensemble.py'; fi

echo "########"
echo "test_GPy"
echo "########"
coverage run 'tests/classifiers/test_GPy.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_GPy.py'; fi

echo "##########"
echo "test_keras"
echo "##########"
coverage run 'tests/classifiers/test_keras.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_keras.py'; fi

echo "#############"
echo "test_keras_tf"
echo "#############"
coverage run 'tests/classifiers/test_keras_tf.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_keras_tf.py'; fi

echo "#############"
echo "test_lightgbm"
echo "#############"
coverage run 'tests/classifiers/test_lightgbm.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_lightgbm.py'; fi

echo "##########"
echo "test_mxnet"
echo "##########"
coverage run 'tests/classifiers/test_mxnet.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_mxnet.py'; fi

echo "############"
echo "test_pytorch"
echo "############"
coverage run 'tests/classifiers/test_pytorch.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_pytorch.py'; fi

echo "################"
echo "test_scikitlearn"
echo "################"
coverage run 'tests/classifiers/test_scikitlearn.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_scikitlearn.py'; fi

echo "###############"
echo "test_tensorflow"
echo "###############"
coverage run 'tests/classifiers/test_tensorflow.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_tensorflow.py'; fi

echo "##################"
echo "test_tensorflow_v2"
echo "##################"
coverage run 'tests/classifiers/test_tensorflow_v2.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_tensorflow_v2.py'; fi

echo "############"
echo "test_xgboost"
echo "############"
coverage run 'tests/classifiers/test_xgboost.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_xgboost.py'; fi

# defences

echo "########################"
echo "test_adversarial_trainer"
echo "########################"
coverage run 'tests/defences/test_adversarial_trainer.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_adversarial_trainer.py'; fi

echo "######################"
echo "test_feature_squeezing"
echo "######################"
coverage run 'tests/defences/test_feature_squeezing.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_feature_squeezing.py'; fi

echo "##########################"
echo "test_gaussian_augmentation"
echo "##########################"
coverage run 'tests/defences/test_gaussian_augmentation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_gaussian_augmentation.py'; fi

echo "#####################"
echo "test_jpeg_compression"
echo "#####################"
coverage run 'tests/defences/test_jpeg_compression.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_jpeg_compression.py'; fi

echo "####################"
echo "test_label_smoothing"
echo "####################"
coverage run 'tests/defences/test_label_smoothing.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_label_smoothing.py'; fi

echo "#################"
echo "test_pixel_defend"
echo "#################"
coverage run 'tests/defences/test_pixel_defend.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_pixel_defend.py'; fi

echo "######################"
echo "test_spatial_smoothing"
echo "######################"
coverage run 'tests/defences/test_spatial_smoothing.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_spatial_smoothing.py'; fi

echo "#########################"
echo "test_thermometer_encoding"
echo "#########################"
coverage run 'tests/defences/test_thermometer_encoding.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_thermometer_encoding.py'; fi

echo "##########################"
echo "test_variance_minimization"
echo "##########################"
coverage run 'tests/defences/test_variance_minimization.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_variance_minimization.py'; fi

# detection

echo "############################"
echo "subsetscanning/test_detector"
echo "############################"
coverage run 'tests/detection/subsetscanning/test_detector.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_detector.py'; fi

echo "#############"
echo "test_detector"
echo "#############"
coverage run 'tests/detection/test_detector.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_detector.py'; fi

# metrics

echo "############"
echo "test_metrics"
echo "############"
coverage run 'tests/metrics/test_metrics.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_metrics.py'; fi

echo "################################"
echo "test_verification_decision_trees"
echo "################################"
coverage run 'tests/metrics/test_verification_decision_trees.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_verification_decision_trees.py'; fi

# poison detection

echo "#######################"
echo "test_activation_defence"
echo "#######################"
coverage run 'tests/poison_detection/test_activation_defence.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_activation_defence.py'; fi

echo "########################"
echo "test_clustering_analyzer"
echo "########################"
coverage run 'tests/poison_detection/test_clustering_analyzer.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_clustering_analyzer.py'; fi

echo "###########################"
echo "test_ground_truth_evaluator"
echo "###########################"
coverage run 'tests/poison_detection/test_ground_truth_evaluator.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_ground_truth_evaluator.py'; fi

echo "#######################"
echo "test_provenance_defence"
echo "#######################"
coverage run 'tests/poison_detection/test_provenance_defence.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_provenance_defence.py'; fi

echo "#########"
echo "test_roni"
echo "#########"
coverage run 'tests/poison_detection/test_roni.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_roni.py'; fi

# wrappers

echo "################"
echo "test_expectation"
echo "################"
coverage run 'tests/wrappers/test_expectation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_expectation.py'; fi

echo "############################"
echo "test_output_add_random_noise"
echo "############################"
coverage run 'tests/wrappers/test_output_add_random_noise.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_add_random_noise.py'; fi

echo "########################"
echo "test_output_class_labels"
echo "########################"
coverage run 'tests/wrappers/test_output_class_labels.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_class_labels.py'; fi

echo "###########################"
echo "test_output_high_confidence"
echo "###########################"
coverage run 'tests/wrappers/test_output_high_confidence.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_high_confidence.py'; fi

echo "###########################"
echo "test_output_reverse_sigmoid"
echo "###########################"
coverage run 'tests/wrappers/test_output_reverse_sigmoid.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_reverse_sigmoid.py'; fi

echo "###################"
echo "test_output_rounded"
echo "###################"
coverage run 'tests/wrappers/test_output_rounded.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_rounded.py'; fi

echo "#######################"
echo "test_query_efficient_bb"
echo "#######################"
coverage run 'tests/wrappers/test_query_efficient_bb.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_query_efficient_bb.py'; fi

echo "#########################"
echo "test_randomized_smoothing"
echo "#########################"
coverage run 'tests/wrappers/test_randomized_smoothing.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_randomized_smoothing.py'; fi

echo "############"
echo "test_wrapper"
echo "############"
coverage run 'tests/wrappers/test_wrapper.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_wrapper.py'; fi

# generators

echo "####################"
echo "test_data_generators"
echo "####################"
coverage run 'tests/test_data_generators.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_data_generators'; fi
codecov

# utils

echo "##########"
echo "test_utils"
echo "##########"
coverage run 'tests/test_utils.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_utils'; fi
codecov

# visualization

echo "##################"
echo "test_visualization"
echo "##################"
coverage run 'tests/test_visualization.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_visualization'; fi

exit $exit_code
