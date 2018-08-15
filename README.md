# Adversarial Robustness Toolbox (ART v0.3.0)
[![Build Status](https://travis-ci.org/IBM/adversarial-robustness-toolbox.svg?branch=master)](https://travis-ci.org/IBM/adversarial-robustness-toolbox) [![GitHub version](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox.svg)](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox) [![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest)

This is a library dedicated to **adversarial machine learning**. Its purpose is to allow rapid crafting and analysis of attacks and defense methods for machine learning models. The Adversarial Robustness Toolbox provides an implementation for many state-of-the-art methods for attacking and defending classifiers.

The library is still under development. Feedback, bug reports and extensions are highly appreciated. Get in touch with us on [Slack](https://ibm-art.slack.com) (invite [here](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTlkMWY3MzgyZDA4ZDdiNzUzY2NhMjc5YmFhZTYzZGYwNDM4YTE1ODhhNDYyNmFlMGFjNWY4ODgyM2EwYTFjYTc))!

## Supported attack and defense methods

The library contains implementations of the following attacks:
* DeepFool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Fast Gradient Method ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))
* Basic Iterative Method ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))
* Jacobian Saliency Map ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))
* Universal Perturbation ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1610.08401))
* Virtual Adversarial Method ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))
* C&amp;W Attack ([Carlini and Wagner, 2016](https://arxiv.org/abs/1608.04644))
* NewtonFool ([Jang et al., 2017](http://doi.acm.org/10.1145/3134600.3134635))

The following defense methods are also supported:
* Feature squeezing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* Spatial smoothing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* Label smoothing ([Warde-Farley and Goodfellow, 2016](https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf))
* Adversarial training ([Szegedy et al., 2013](http://arxiv.org/abs/1312.6199))
* Virtual adversarial training ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))
* Gaussian data augmentation ([Zantedeschi et al., 2017](https://arxiv.org/abs/1707.06728))

## Setup

### Installation with `pip`

The toolbox is designed to run with Python 2 and 3.
The library can be installed from the PyPi repository using `pip`:

```bash
pip install adversarial-robustness-toolbox
```

### Manual installation

For the most recent version of the library, either download the source code or clone the repository in your directory of choice:

```bash
git clone https://github.com/IBM/adversarial-robustness-toolbox
```

To install ART, do the following in the project folder:
```bash
pip install .
```

The library comes with a basic set of unit tests. To check your install, you can run all the unit tests by calling the test script in the install folder:

```bash
bash run_tests.sh
```

## Running ART

Some examples of how to use ART when writing your own code can be found in the `examples` folder. See `examples/README.md` for more information about what each example does. To run an example, use the following command:
```bash
python examples/<example_name>.py
```

The `notebooks` folder contains Jupyter notebooks with detailed walkthroughs of some usage scenarios. 

## Citing ART

If you use ART for research, please consider citing the following reference paper:
```
@article{art2018,
    title = {Adversarial Robustness Toolbox v0.3.0},
    author = {Nicolae, Maria-Irina and Sinn, Mathieu and Tran, Minh~Ngoc and Rawat, Ambrish and Wistuba, Martin and Zantedeschi, Valentina and Baracaldo, Nathalie and Chen, Bryant and Ludwig, Heiko and Molloy, Ian and Edwards, Ben},
    journal = {CoRR},
    volume = {1807.01069}
    year = {2018},
    url = {https://arxiv.org/pdf/1807.01069}
}
```
