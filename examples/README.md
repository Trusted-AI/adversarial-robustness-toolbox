# Adversarial Robustness Toolbox examples

## Get Started with ART
These examples train a small model on the MNIST dataset and creates adversarial examples using the Fast Gradient Sign 
Method. Here we use the ART classifier to train the model, it would also be possible to provide a pretrained model to 
the ART classifier. The parameters are chosen for reduced computational requirements of the script and not optimised 
for accuracy.


### TensorFlow
[get_started_tensorflow_v2.py](get_started_tensorflow_v2.py) demonstrates a simple example of using ART with TensorFlow v2.x. 

### Keras
[get_started_keras.py](get_started_keras.py) demonstrates a simple example of using ART with Keras.

### PyTorch
[get_started_pytorch.py](get_started_pytorch.py) demonstrates a simple example of using ART with PyTorch.

### Scikit-learn
[get_started_scikit_learn.py](get_started_scikit_learn.py) demonstrates a simple example of using ART with Scikit-learn. 
This example uses the support vector machine SVC, but any other classifier of Scikit-learn can be used as well.

### XGBoost
[get_started_xgboost.py](get_started_xgboost.py) demonstrates a simple example of using ART with XGBoost.
Because gradient boosted tree classifier do not provide gradients, the adversarial examples are created with the 
black-box method Zeroth Order Optimization.

### InverseGAN
[get_started_inverse_gan.py](get_started_inverse_gan.py) demonstrates a simple example of using 
InverseGAN and Defense ART with TensorFlow v1.x.

### LightGBM
[get_started_lightgbm.py](get_started_lightgbm.py) demonstrates a simple example of using ART with LightGBM.
Because gradient boosted tree classifier do not provide gradients, the adversarial examples are created with the 
black-box method Zeroth Order Optimization.


## Applications

[adversarial_training_cifar10.py](adversarial_training_cifar10.py) trains a convolutional neural network on the CIFAR-10 
dataset, then generates adversarial images using the DeepFool attack and retrains the network on the training set 
augmented with the adversarial images.

[adversarial_training_data_augmentation.py](adversarial_training_data_augmentation.py) shows how to use ART and Keras 
to perform adversarial training using data generators for CIFAR-10.

[mnist_cnn_fgsm.py](mnist_cnn_fgsm.py) trains a convolutional neural network on MNIST, then crafts FGSM attack examples 
on it.

[mnist_poison_detection.py](mnist_poison_detection.py) generates a backdoor for MNIST dataset, then trains a 
convolutional neural network on the poisoned dataset and runs activation defence to find poison.

### Fast Minimum-Norm (FMN) on handwritten digits

[fast_minimum_norm_digits.py](fast_minimum_norm_digits.py) evaluates the published
[FMN attack (Pintor et al., NeurIPS 2021)](https://arxiv.org/abs/2102.12827) with L2 and Linf norms, in targeted
and untargeted modes. It trains a small tanh MLP on a fixed training split and attacks the first 100 correctly
classified held-out samples. The real 8×8 digit images come from scikit-learn's bundled `load_digits` dataset;
no dataset download is needed. The example runs on CPU and requires ART, scikit-learn and PyTorch.

Run from the repository root after installing these packages:

```bash
python examples/fast_minimum_norm_digits.py --samples 100 --steps 100 --output-dir /tmp/fmn-digits
```

The output folder contains `results.json` (clean accuracy, attack success counts and median successful perturbation
norms) and `samples.npz` (exact inputs, adversarial outputs, labels and split indices). Failed searches return the
original sample; these failures count against the success rate and are excluded from the successful-norm median.
Results are a small evaluation, not a reproduction of the paper's full benchmarks or a fixed-epsilon robustness score.
Use `--model linear` to evaluate a logistic-regression model instead of the default MLP.

To compare with the [authors' original implementation](https://github.com/pralab/Fast-Minimum-Norm-FMN-Attack),
install Foolbox and EagerPy and pass `--reference-source /path/to/src/attacks/fmn.py`. The example runs that source
on the same model and samples, and records its SHA-256 hash and comparison metrics. Reference code is not downloaded
automatically.
