# Nemesis (v0.1)
This is a library dedicated to **adversarial machine learning.** Its purpose is to allow rapid crafting and analysis of attacks and defense methods for machine learning models. Nemesis provides an implementation for the majority of state-of-the-art methods for attacking classifiers. 

The library is still under development. Feedback, bug reports and extension requests are highly appreciated.

## Supported attack methods

Nemesis contains implementations of the following attacks:
* Deep Fool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Fast Gradient Method ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))
* Jacobian Saliency Map ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))
* Universal Perturbation ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1610.08401))
* Virtual Adversarial Method ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1507.00677))

## Setup

### Requirements

Nemesis is designed to run with Python 3 (and most likely Python 2 with small changes). You can either download the source code of Nemesis or clone the repository in your directory of choice:
```bash
git clone https://github.ibm.com/Maria-Irina-Nicolae/nemesis
```

To install the project dependencies, use the requirements file:
```bash
pip install -r requirements.txt
```

You will additionally need to download [Cleverhans](https://github.com/tensorflow/cleverhans).

### Installation

Nemesis is linked against Cleverhans through the configuration file `config/config.ini`. When installing Nemesis on your local machine, you need to set the appropriate paths and the `LOCAL` configuration profile as follows:

```text
[DEFAULT]
profile=LOCAL

[LOCAL]
data_path=/local/path/here
mnist_path=/local/path/here
cifar10_path=/local/path/here
cleverhans_path=/local/path/here
```

If the Mnist and Cifar-10 datasets are not present at the indicated path, loading them will also download the data.

The library comes with a basic set of unit tests. To check that the installation has succeeded, you can run all the unit tests by calling in the Nemesis folder:
```bash
bash tests.sh
```

## Running Nemesis

The library contains three main scripts for:
* training a classifier using (`train.py`)
* crafting adversarial examples on a trained model through (`generate_adversaral.py`)
* testing model accuracy on different test sets using (`test_accuracies.py`)

Detailed instructions for each script are available by typing
```python
python3 <script_name> -h
```

