Tracking the tests for huggingface integration.

These have been explicitly checked and re-factored. Other pytest checks may be running HF incidentally by virtue of 
not excluding it. Unittest based checks will not be checking for HF.

+ defences/trainer/all
    + test_adversarial_trainer
    + test_adversarial_trainer_FBF
    + test_adversarial_trainer_madry_pgd
    + test_adversarial_trainer_trades_pytorch
    + test_dp_instahide_trainer
+ defences/poison/pytest/test_activation_defence_pytest
+ defences/poison/test_spectral_signature_defense

General
+ estimators/classification/deeplearning_common

Attacks
+ attacks/poison/test_clean_label_backdoor_attack
+ attacks/poison/test_hidden_trigger_backdoor
+ attacks/poison/test_backdoor_attack_dgm_red
+ attacks/poison/test_backdoor_attack_dgm_trail
+ attacks/evasion/test_projected_gradient_descent