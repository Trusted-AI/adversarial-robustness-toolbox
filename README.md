# Nemesis (v0.2.2)

This is a library dedicated to **adversarial machine learning**. Its purpose is to allow rapid crafting and analysis of attacks and defense methods for machine learning models. Nemesis provides an implementation for many state-of-the-art methods for attacking and defending classifiers.

The library is still under development. Feedback, bug reports and extensions are highly appreciated.

## Supported attack and defense methods

Nemesis contains implementations of the following attacks:
* DeepFool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Fast Gradient Method ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))
* Basic Iterative Method ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))
* Jacobian Saliency Map ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))
* Universal Perturbation ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1610.08401))
* Virtual Adversarial Method ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1507.00677))
* C&amp;W Attack ([Carlini and Wagner, 2016](https://arxiv.org/abs/1608.04644))
* NewtonFool ([Jang et al., 2017](http://doi.acm.org/10.1145/3134600.3134635))

The following defense methods are also supported:
* Feature squeezing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* Spatial smoothing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* Label smoothing ([Warde-Farley and Goodfellow, 2016](https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf))
* Adversarial training ([Szegedy et al., 2013](http://arxiv.org/abs/1312.6199))
* Virtual adversarial training ([Miyato et al., 2017](https://arxiv.org/abs/1704.03976))
* Gaussian data augmentation ([Zantedeschi et al., 2017](https://arxiv.org/abs/1707.06728))

## Setup

Nemesis is designed to run with Python 3 (and most likely Python 2 with small changes). You can either download the source code of Nemesis or clone the repository in your directory of choice:
```bash
git clone https://github.ibm.com/Maria-Irina-Nicolae/nemesis
```

To install the project dependencies, use the requirements file:
```bash
pip install .
```

The library comes with a basic set of unit tests. To check your install, you can run all the unit tests by calling in the Nemesis folder:
```bash
bash run_tests.sh
```

The configuration file `config/config.ini` allows to set custom paths for data. By default, data is downloaded in the `nemesis/data` folder as follows:

```text
[DEFAULT]
profile=LOCAL

[LOCAL]
data_path=./data
mnist_path=./data/mnist
cifar10_path=./data/cifar-10
stl10_path=./data/stl-10
```

If the datasets are not present at the indicated path, loading them will also download the data.

## Running Nemesis

Some examples of how to use Nemesis when writing your own code can be found in the `examples` folder. See `examples/README.md` for more information about what each example does. To run an example, use the following command:
```bash
python3 examples/<example_name>.py
```
