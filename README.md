# Adversarial Robustness Toolbox (ART v0.1)
[![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest)

This is a library dedicated to **adversarial machine learning**. Its purpose is to allow rapid crafting and analysis of attacks and defense methods for machine learning models. The Adversarial Robustness Toolbox provides an implementation for many state-of-the-art methods for attacking and defending classifiers.

The library is still under development. Feedback, bug reports and extensions are highly appreciated. Get in touch with us on [Slack](https://ibm-art.slack.com) (invite [here]( https://join.slack.com/t/ibm-art/shared_invite/enQtMzczOTgyODUyMzU1LTFmNmI1NmM5Mjk4YjdjOTc0ZjU1ODQ5MTFlMzFhNjE3MDc5ZDFmYmQzNDZjMjY4ZDA4NjU2Yjk2MmQ4OGVhMDg))!

## Supported attack and defense methods

The Adversarial Robustness Toolbox contains implementations of the following attacks:
* Deep Fool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Fast Gradient Method ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))
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

## Setup

The Adversarial Robustness Toolbox is designed to run with Python 3 (and most likely Python 2 with small changes). You can either download the source code or clone the repository in your directory of choice:
```bash
git clone https://github.com/IBM/adversarial-robustness-toolbox
```

To install the project dependencies, use the requirements file:
```bash
pip install .
```

The library comes with a basic set of unit tests. To check your install, you can run all the unit tests by calling in the library folder:
```bash
bash run_tests.sh
```

The configuration file `config/config.ini` allows to set custom paths for data. By default, data is downloaded in the `data` folder as follows:

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

## Running ART

Some examples of how to use ART when writing your own code can be found in the `examples` folder. See `examples/README.md` for more information about what each example does. To run an example, use the following command:

```bash
python3 examples/<example_name>.py
```
