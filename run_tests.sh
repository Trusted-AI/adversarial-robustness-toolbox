#!/usr/bin/env bash

exit_code=0

# Set TensorFlow logging to minimum level ERROR
export TF_CPP_MIN_LOG_LEVEL="3"

# --------------------------------------------------------------------------------------------------------------- TESTS

# NOTE: All the tests should be ran within this loop. All other tests are legacy tests that must be made framework
# independent to be incorporated within this loop
frameworkList=("tensorflow" "keras" "pytorch" "scikitlearn" "mxnet" "kerastf")
framework=$1
legacy_module=$2

if [[ ${framework} != "legacy" ]]
then
    echo "#######################################################################"
    echo "############### Running tests with framework $framework ###############"
    echo "#######################################################################"

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/defences/detector/evasion --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed defences/detector/evasion tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/defences/detector/poison/test_spectral_signature_defense.py --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed defences/detector/poison/test_spectral_signature_defense.py tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/defences/preprocessor --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed defences/preprocessor tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/defences/trainer --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed defences/trainer tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/defences/transformer --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed defences/transformer tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/preprocessing/audio --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed preprocessing/audio tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/preprocessing/expectation_over_transformation --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed preprocessing/expectation_over_transformation tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/utils --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed utils tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv -s tests/attacks/poison/ --framework=$framework  --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed attacks/poison tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv -s tests/attacks/evasion/ --framework=$framework  --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed attacks/evasion"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/estimators/speech_recognition/ --framework=$framework  --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/speech_recognition tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/attacks/inference/ --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed attacks/inference"; fi

    pytest --cov-report=xml --cov=art --cov-append -q -vv tests/classifiersFrameworks/  --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed classifiersFrameworks tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/estimators/classification/test_deeplearning_common.py --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/classification/test_deeplearning_common.py $framework"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/estimators/classification/test_deeplearning_specific.py --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/classification tests for framework $framework"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/estimators/certification/ --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/certification tests for framework $framework"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/estimators/classification/test_blackbox_existing_predictions.py --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/classification/test_blackbox_existing_predictions.py $framework"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/estimators/regression/test_blackbox.py --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/regression/test_blackbox.py $framework"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/metrics/privacy --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed metrics/privacy tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/evaluations --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed evaluations tests"; fi

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/test_summary_writer.py --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed summary writer tests"; fi

else
    declare -a attacks_1=("tests/attacks/test_adversarial_patch.py" \
                          "tests/attacks/test_adversarial_embedding.py" \
                          "tests/attacks/test_backdoor_attack.py" \
                          "tests/attacks/test_carlini.py" \
                          "tests/attacks/test_copycat_cnn.py" \
                          "tests/attacks/test_decision_tree_attack.py" \
                          "tests/attacks/test_deepfool.py" \
                          "tests/attacks/test_elastic_net.py" \
                          "tests/attacks/test_feature_collision.py" \
                          "tests/attacks/test_functionally_equivalent_extraction.py" \
                          "tests/attacks/test_graphite.py" \
                          "tests/attacks/test_hclu.py" \
                          "tests/attacks/test_input_filter.py" \
                          "tests/attacks/test_hop_skip_jump.py" \
                          "tests/attacks/test_iterative_method.py" \
                          "tests/attacks/test_knockoff_nets.py" )

    declare -a attacks_2=("tests/attacks/test_newtonfool.py" \
                          "tests/attacks/test_poisoning_attack_svm.py" \
                          "tests/attacks/test_projected_gradient_descent.py" \
                          "tests/attacks/test_saliency_map.py" \
                          "tests/attacks/test_spatial_transformation.py" \
                          "tests/attacks/test_universal_perturbation.py" \
                          "tests/attacks/test_virtual_adversarial.py" \
                          "tests/attacks/test_zoo.py" \
                          "tests/attacks/test_pixel_attack.py" \
                          "tests/attacks/test_threshold_attack.py" \
                          "tests/attacks/test_wasserstein.py" \
                          "tests/attacks/test_shapeshifter.py" \
                          "tests/attacks/test_targeted_universal_perturbation.py" \
                          "tests/attacks/test_simba.py" )

    declare -a estimators=("tests/estimators/certification/test_randomized_smoothing.py" \
                           "tests/estimators/certification/test_derandomized_smoothing.py" \
                           "tests/estimators/classification/test_blackbox.py" \
                           "tests/estimators/classification/test_catboost.py" \
                           "tests/estimators/classification/test_classifier.py" \
                           "tests/estimators/classification/test_deep_partition_ensemble.py" \
                           "tests/estimators/classification/test_detector_classifier.py" \
                           "tests/estimators/classification/test_ensemble.py" \
                           "tests/estimators/classification/test_GPy.py" \
                           "tests/estimators/classification/test_input_filter.py" \
                           "tests/estimators/classification/test_lightgbm.py" \
                           "tests/estimators/classification/test_query_efficient_bb.py" \
                           "tests/estimators/classification/test_scikitlearn.py" \
                           "tests/estimators/classification/test_xgboost.py" \
                           "tests/estimators/regression/test_scikitlearn.py" )

    declare -a defences=("tests/defences/test_adversarial_trainer.py" \
                         "tests/defences/test_adversarial_trainer_madry_pgd.py" \
                         "tests/defences/test_class_labels.py" \
                         "tests/defences/test_defensive_distillation.py" \
                         "tests/defences/test_feature_squeezing.py" \
                         "tests/defences/test_gaussian_augmentation.py" \
                         "tests/defences/test_gaussian_noise.py" \
                         "tests/defences/test_high_confidence.py" \
                         "tests/defences/test_label_smoothing.py" \
                         "tests/defences/test_neural_cleanse.py" \
                         "tests/defences/test_pixel_defend.py" \
                         "tests/defences/test_reverse_sigmoid.py" \
                         "tests/defences/test_rounded.py" \
                         "tests/defences/test_thermometer_encoding.py" \
                         "tests/defences/test_variance_minimization.py" \
                         "tests/defences/detector/poison/test_activation_defence.py" \
                         "tests/defences/detector/poison/test_clustering_analyzer.py" \
                         "tests/defences/detector/poison/test_ground_truth_evaluator.py" \
                         "tests/defences/detector/poison/test_provenance_defence.py" \
                         "tests/defences/detector/poison/test_roni.py" )

    declare -a metrics=("tests/metrics/test_gradient_check.py" \
                        "tests/metrics/test_metrics.py" \
                        "tests/metrics/test_verification_decision_trees.py" )

    declare -a art=("tests/test_data_generators.py" \
                    "tests/test_optimizers.py" \
                    "tests/test_utils.py" \
                    "tests/test_visualization.py" )

    # ----------------------------------------------------------------------------------------------- CODE TO RUN TESTS

    run_test () {
      test=$1
      test_file_name="$(echo ${test} | rev | cut -d'/' -f1 | rev)"

      echo $'\n\n'
      echo "######################################################################"
      echo ${test}
      echo "######################################################################"
      pytest --cov-report=xml --cov=art --cov-append  -q -vv ${test} --durations=0
      if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed $test"; fi
    }

    tests="$legacy_module[@]"
    for test in "${!tests}"; do
        run_test ${test}
    done
fi

exit ${exit_code}
