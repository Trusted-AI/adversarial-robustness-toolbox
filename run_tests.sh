#!/usr/bin/env bash
exit_code=0

# attacks
python -m unittest 'tests/attacks/test_adversarial_patch.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_adversarial_patch.py'; fi

python -m unittest 'tests/attacks/test_boundary.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_boundary.py'; fi

python -m unittest 'tests/attacks/test_carlini.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_carlini.py'; fi

python -m unittest 'tests/attacks/test_decision_tree_attack.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_decision_tree_attack.py'; fi

python -m unittest 'tests/attacks/test_deepfool.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_deepfool.py'; fi

python -m unittest 'tests/attacks/test_elastic_net.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_elastic_net.py'; fi

python -m unittest 'tests/attacks/test_fast_gradient.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_fast_gradient.py'; fi

python -m unittest 'tests/attacks/test_functionally_equivalent_extraction.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_functionally_equivalent_extraction.py'; fi

python -m unittest 'tests/attacks/test_hclu.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_hclu.py'; fi

python -m unittest 'tests/attacks/test_hop_skip_jump.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_hop_skip_jump.py'; fi

python -m unittest 'tests/attacks/test_input_filter.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_input_filter.py'; fi

python -m unittest 'tests/attacks/test_iterative_method.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_iterative_method.py'; fi

python -m unittest 'tests/attacks/test_newtonfool.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_newtonfool.py'; fi

python -m unittest 'tests/attacks/test_poisoning_attack_svm.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_poisoning_attack_svm.py'; fi

python -m unittest 'tests/attacks/test_projected_gradient_descent.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_projected_gradient_descent.py'; fi

python -m unittest 'tests/attacks/test_saliency_map.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_saliency_map.py'; fi

python -m unittest 'tests/attacks/test_spatial_transformation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_spatial_transformation.py'; fi

python -m unittest 'tests/attacks/test_universal_perturbation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_universal_perturbation.py'; fi

python -m unittest 'tests/attacks/test_virtual_adversarial.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_virtual_adversarial.py'; fi

python -m unittest 'tests/attacks/test_zoo.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_zoo.py'; fi

# classifiers
python -m unittest 'tests/classifiers/test_blackbox.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_blackbox.py'; fi

python -m unittest 'tests/classifiers/test_catboost.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_catboost.py'; fi

python -m unittest 'tests/classifiers/test_classifier.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_classifier.py'; fi

python -m unittest 'tests/classifiers/test_detector_classifier.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_detector_classifier.py'; fi

python -m unittest 'tests/classifiers/test_ensemble.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_ensemble.py'; fi

python -m unittest 'tests/classifiers/test_GPy.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_GPy.py'; fi

python -m unittest 'tests/classifiers/test_input_filter.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_input_filter.py'; fi

python -m unittest 'tests/classifiers/test_keras.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_keras.py'; fi

python -m unittest 'tests/classifiers/test_keras_tf.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_keras_tf.py'; fi

python -m unittest 'tests/classifiers/test_lightgbm.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_lightgbm.py'; fi

python -m unittest 'tests/classifiers/test_mxnet.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_mxnet.py'; fi

python -m unittest 'tests/classifiers/test_pytorch.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_pytorch.py'; fi

python -m unittest 'tests/classifiers/test_scikitlearn.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_scikitlearn.py'; fi

python -m unittest 'tests/classifiers/test_tensorflow.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_tensorflow.py'; fi

python -m unittest 'tests/classifiers/test_tensorflow_v2.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_tensorflow_v2.py'; fi

python -m unittest 'tests/classifiers/test_xgboost.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_xgboost.py'; fi

# defences
python -m unittest discover tests/defences -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed tests/defences'; fi

# detection
python -m unittest discover tests/detection -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed tests/detection'; fi

# poison detection
python -m unittest discover tests/poison_detection -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed tests/poison_detection'; fi

# wrappers
python -m unittest 'tests/wrappers/test_expectation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_expectation.py'; fi

python -m unittest 'tests/wrappers/test_output_add_random_noise.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_add_random_noise.py'; fi

python -m unittest 'tests/wrappers/test_output_class_labels.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_class_labels.py'; fi

python -m unittest 'tests/wrappers/test_output_high_confidence.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_high_confidence.py'; fi

python -m unittest 'tests/wrappers/test_output_reverse_sigmoid.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_reverse_sigmoid.py'; fi

python -m unittest 'tests/wrappers/test_output_rounded.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_output_rounded.py'; fi

python -m unittest 'tests/wrappers/test_query_efficient_bb.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_query_efficient_bb.py'; fi

python -m unittest 'tests/wrappers/test_randomized_smoothing.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_randomized_smoothing.py'; fi

python -m unittest 'tests/wrappers/test_expectation.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_wrapper.py'; fi

# generators
python -m unittest tests.test_data_generators
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_data_generators'; fi

# metrics
python -m unittest discover tests/metrics -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed tests/metrics'; fi

# utils
python -m unittest tests.test_utils
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_utils'; fi

# visualization
python -m unittest tests.test_visualization
if [[ $? -ne 0 ]]; then exit_code=1; echo 'Failed test_visualization'; fi

exit $exit_code
