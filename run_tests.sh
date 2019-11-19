#!/usr/bin/env bash
exit_code=0

# attacks
python -m unittest 'tests/attacks/test_adversarial_patch.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_boundary.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_carlini.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_decision_tree_attack.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_deepfool.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_elastic_net.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_fast_gradient.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_hclu.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_hop_skip_jump.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_iterative_method.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_newtonfool.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_poisoning_attack_svm.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_projected_gradient_descent.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_saliency_map.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_spatial_transformation.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_universal_perturbation.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_virtual_adversarial.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/attacks/test_zoo.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi

# classifiers
python -m unittest 'tests/classifiers/test_blackbox.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_catboost.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_classifier.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_detector_classifier.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_ensemble.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_GPy.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_keras.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_keras_tf.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_lightgbm.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_mxnet.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_pytorch.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_scikitlearn.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_tensorflow.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_tensorflow_v2.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest 'tests/classifiers/test_xgboost.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi

# defences
python -m unittest discover tests/defences -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi

# detection
python -m unittest discover tests/detection -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi

# poison detection
python -m unittest discover tests/poison_detection -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi

# wrappers
python -m unittest discover tests/wrappers -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi

# generators
python -m unittest tests.test_data_generators
if [[ $? -ne 0 ]]; then exit_code=1; fi

# metrics
python -m unittest discover tests/metrics -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi

# utils
python -m unittest tests.test_utils
if [[ $? -ne 0 ]]; then exit_code=1; fi

# visualization
python -m unittest tests.test_visualization
if [[ $? -ne 0 ]]; then exit_code=1; fi

exit $exit_code