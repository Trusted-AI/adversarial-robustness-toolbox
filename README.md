# Nemesis (v1.0.0)
This is a library dedicated to **adversarial machine learning**. Its purpose is to allow rapid crafting and analysis of attacks and defense methods for machine learning models. Nemesis provides an implementation for many state-of-the-art methods for attacking and defending classifiers.

The library is still under development. Feedback, bug reports and extension requests are highly appreciated.

## Supported attack and defense methods

Nemesis contains implementations of the following attacks:
* Deep Fool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Fast Gradient Method ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))
* Jacobian Saliency Map ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))
* Universal Perturbation ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1610.08401))
* Virtual Adversarial Method ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1507.00677))

The following defense methods are also supported:
* Feature squeezing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* Label smoothing (Warde-Farley and Goodfellow, 2016)
* Adversarial training ([Szegedy et al., 2013](http://arxiv.org/abs/1312.6199))
* Virtual adversarial training ([Miyato et al., 2017](https://arxiv.org/abs/1704.03976))

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
stl10_path=/local/path/here
cleverhans_path=/local/path/here
```

If the datasets are not present at the indicated path, loading them will also download the data.

The library comes with a basic set of unit tests. To check that the installation has succeeded, you can run all the unit tests by calling in the Nemesis folder:
```bash
bash run_tests.sh
```

## Running Nemesis

The library contains three main scripts for:
* training a classifier using (`train.py`)
* crafting adversarial examples on a trained model through (`generate_adversarial.py`)
* testing model accuracy on different test sets using (`test_accuracies.py`)

Detailed instructions for each script are available by typing
```python
python3 <script_name> -h
```

Some examples of how to use Nemesis when writing your own code can be found in the `examples` folder. See `examples/README.md` for more information about what each example does. To run an example, use the following command:
```bash
python3 examples/<example_name>.py
```