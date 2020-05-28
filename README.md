# Adversarial Robustness Toolbox (ART) v1.2
<p align="center">
  <img src="docs/images/art_logo.png?raw=true" width="200" title="ART logo">
</p>
<br />

[![Build Status](https://travis-ci.org/IBM/adversarial-robustness-toolbox.svg?branch=master)](https://travis-ci.org/IBM/adversarial-robustness-toolbox)
[![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest)
[![GitHub version](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox.svg)](https://badge.fury.io/gh/IBM%2Fadversarial-robustness-toolbox)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IBM/adversarial-robustness-toolbox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/adversarial-robustness-toolbox/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IBM/adversarial-robustness-toolbox.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/adversarial-robustness-toolbox/alerts/)
[![codecov](https://codecov.io/gh/IBM/adversarial-robustness-toolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/IBM/adversarial-robustness-toolbox)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adversarial-robustness-toolbox)](https://pypi.org/project/adversarial-robustness-toolbox/)
[![slack-img](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://ibm-art.slack.com/)

[中文README请按此处](README-cn.md)

Adversarial Robustness Toolbox (ART) is a Python library supporting developers and researchers in defending Machine 
Learning models (Deep Neural Networks, Gradient Boosted Decision Trees, Support Vector Machines, Random Forests, 
Logistic Regression, Gaussian Processes, Decision Trees, Scikit-learn Pipelines, etc.) against adversarial threats 
(including evasion, extraction and poisoning) and helps making AI systems more secure and trustworthy. Machine Learning 
models are vulnerable to adversarial examples, which are inputs (images, texts, tabular data, etc.) deliberately crafted 
to produce a desired response by the Machine Learning model. ART provides the tools to build and deploy defences and 
test them with adversarial attacks. 

Defending Machine Learning models involves certifying and verifying model robustness and model hardening with 
approaches such as pre-processing inputs, augmenting training data with adversarial examples, and leveraging runtime 
detection methods to flag any inputs that might have been modified by an adversary. ART includes attacks for testing 
defenses with state-of-the-art threat models.

Documentation of ART: https://adversarial-robustness-toolbox.readthedocs.io

Get started with [examples](examples/README.md) and [tutorials](notebooks/README.md)

The library is under continuous development. Feedback, bug reports and contributions are very welcome. 
Get in touch with us on [Slack](https://ibm-art.slack.com) (invite [here](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTA4NGQ1OTMxMzFmY2Q1MzE1NWI2MmEzN2FjNGNjOGVlODVkZDE0MjA1NTA4OGVkMjVkNmQ4MTY1NmMyOGM5YTg))!

## Supported Machine Learning Libraries and Applications
* TensorFlow (v1 and v2) (www.tensorflow.org)
* Keras (www.keras.io)
* PyTorch (www.pytorch.org)
* MXNet (https://mxnet.apache.org)
* Scikit-learn (www.scikit-learn.org)
* XGBoost (www.xgboost.ai)
* LightGBM (https://lightgbm.readthedocs.io)
* CatBoost (www.catboost.ai)
* GPy (https://sheffieldml.github.io/GPy/)

## Implemented Attacks, Defences, Detections, Metrics, Certifications and Verifications

**Evasion Attacks:**
* Threshold Attack ([Vargas et al., 2019](https://arxiv.org/abs/1906.06026))
* Pixel Attack ([Vargas et al., 2019](https://arxiv.org/abs/1906.06026), [Su et al., 2019](https://ieeexplore.ieee.org/abstract/document/8601309/citations#citations))
* HopSkipJump attack ([Chen et al., 2019](https://arxiv.org/abs/1904.02144))
* High Confidence Low Uncertainty adversarial samples ([Grosse et al., 2018](https://arxiv.org/abs/1812.02606))
* Iterative frame saliency attack ([Inkawhich et al., 2018](https://arxiv.org/abs/1811.11875))
* DPatch ([Liu et al., 2018](https://arxiv.org/abs/1806.02299v4))
* Projected gradient descent ([Madry et al., 2017](https://arxiv.org/abs/1706.06083))
* NewtonFool ([Jang et al., 2017](http://doi.acm.org/10.1145/3134600.3134635))
* Elastic net attack ([Chen et al., 2017](https://arxiv.org/abs/1709.04114))
* Spatial transformation attack ([Engstrom et al., 2017](https://arxiv.org/abs/1712.02779))
* Query-efficient black-box attack ([Ilyas et al., 2017](https://arxiv.org/abs/1712.07113))
* Zeroth-order optimization attack ([Chen et al., 2017](https://arxiv.org/abs/1708.03999))
* Decision-based attack / Boundary attack ([Brendel et al., 2018](https://arxiv.org/abs/1712.04248))
* Adversarial patch ([Brown et al., 2017](https://arxiv.org/abs/1712.09665))
* Decision tree attack ([Papernot et al., 2016](https://arxiv.org/abs/1605.07277))
* Carlini & Wagner (C&W) `L_2` and `L_inf` attacks ([Carlini and Wagner, 2016](https://arxiv.org/abs/1608.04644))
* Basic iterative method ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))
* Jacobian saliency map ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))
* Universal perturbation ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1610.08401))
* Feature Adversaries ([Sabour et al., 2016](https://arxiv.org/abs/1511.05122))
* DeepFool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Virtual adversarial method ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))
* Fast gradient method ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))

**Extraction Attacks:**
* Functionally Equivalent Extraction ([Jagielski et al., 2019](https://arxiv.org/abs/1909.01838))
* Copycat CNN ([Correia-Silva et al., 2018](https://arxiv.org/abs/1806.05476))
* KnockoffNets ([Orekondy et al., 2018](https://arxiv.org/abs/1812.02766))

**Poisoning Attacks:**
* Poisoning Attack on SVM ([Biggio et al., 2013](https://arxiv.org/abs/1206.6389))
* Backdoor Attack ([Gu, et. al., 2017](https://arxiv.org/abs/1708.06733))
* Feature Collision Attack ([Shafahi, Huang et. al., 2018](https://arxiv.org/pdf/1804.00792.pdf))

**Defences - Preprocessor:**
* Resampling ([Yang et al., 2019](https://arxiv.org/abs/1809.10875))
* Thermometer encoding ([Buckman et al., 2018](https://openreview.net/forum?id=S18Su--CW))
* MP3 compression ([Carlini, N. & Wagner, D., 2018](https://arxiv.org/abs/1801.01944))
* Total variance minimization ([Guo et al., 2018](https://openreview.net/forum?id=SyJ7ClWCb))
* PixelDefend ([Song et al., 2017](https://arxiv.org/abs/1710.10766))
* Gaussian data augmentation ([Zantedeschi et al., 2017](https://arxiv.org/abs/1707.06728))
* Feature squeezing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* Spatial smoothing ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* JPEG compression ([Dziugaite et al., 2016](https://arxiv.org/abs/1608.00853))
* Label smoothing ([Warde-Farley and Goodfellow, 2016](https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf))
* Virtual adversarial training ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))

**Defences - Postprocessor:**
* Reverse Sigmoid ([Lee et al., 2018](https://arxiv.org/abs/1806.00054))
* Random Noise ([Chandrasekaranet al., 2018](https://arxiv.org/abs/1811.02054))
* Class Labels ([Tramer et al., 2016](https://arxiv.org/abs/1609.02943), [Chandrasekaranet al., 2018](https://arxiv.org/abs/1811.02054))
* High Confidence ([Tramer et al., 2016](https://arxiv.org/abs/1609.02943))
* Rounding ([Tramer et al., 2016](https://arxiv.org/abs/1609.02943))

**Defences - Trainer:**
* Adversarial training ([Szegedy et al., 2013](http://arxiv.org/abs/1312.6199))
* Adversarial training Madry PGD ([Madry et al., 2017](https://arxiv.org/abs/1706.06083))

**Defences - Transformer:**
* Defensive Distillation ([Papernot et al., 2015](https://arxiv.org/abs/1511.04508))

**Robustness Metrics, Certifications and Verifications**:
* Clique Method Robustness Verification ([Hongge et al., 2019](https://arxiv.org/abs/1906.03849))
* Randomized Smoothing ([Cohen et al., 2019](https://arxiv.org/abs/1902.02918))
* CLEVER ([Weng et al., 2018](https://arxiv.org/abs/1801.10578))
* Loss sensitivity ([Arpit et al., 2017](https://arxiv.org/abs/1706.05394))
* Empirical robustness ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))

**Detection of Adversarial Examples:**
* Basic detector based on inputs
* Detector trained on the activations of a specific layer
* Detector based on Fast Generalized Subset Scan ([Speakman et al., 2018](https://arxiv.org/pdf/1810.08676))

**Detection of Poisoning Attacks:**
* Detection based on activations analysis ([Chen et al., 2018](https://arxiv.org/abs/1811.03728))
* Detection based on data provenance ([Baracaldo et al., 2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8473440))
* Detection based on spectral signatures ([Tran et al., 2018](https://papers.nips.cc/paper/8024-spectral-signatures-in-backdoor-attacks.pdf))

## Setup

### Installation with `pip`

The toolbox is designed and tested to run with Python 3. 
ART can be installed from the PyPi repository using `pip`:

```bash
pip install adversarial-robustness-toolbox
```

### Manual installation
The most recent version of ART can be downloaded or cloned from this repository:

```bash
git clone https://github.com/IBM/adversarial-robustness-toolbox
```

Install ART with the following command from the project folder `adversarial-robustness-toolbox`:

Using pip:
```bash
pip install .
```

Using Docker:
* Build the ART docker image: `make build`
* To enter the ART docker environment run: `make run-bash`
* To run the Jupyter notebooks from the container run `make run-jupyter` and copy and paste the url generated \
to connect to that container.  

#### Running ART Unit tests
ART provides unit tests that can be ran within the ART environment. The first time the tests are ran,
 ART will download the necessary datasets so it might take a while to do so.
* Note: If you wish to run the unit tests using the Tensorflow 1 environment simply comment out the tensorflow 2 \
packages within either the `test_requirements.txt` or `Dockerfile` as needed

Use the following commands to run the tests:

Using pip:
```bash
pip install -r test_requirements.txt
bash run_tests.sh
```

Using Docker:
`make run-test`


## Getting Started with ART

Examples of using ART can be found in `examples` and [examples/README.md](examples/README.md) provides an overview and 
additional information. It contains a minimal example for each machine learning framework. All examples can be run with
the following command:
```bash
python examples/<example_name>.py
```

More detailed examples and tutorials are located in `notebooks` and [notebooks/README.md](notebooks/README.md) provides 
and overview and more information. 


## Contributing to ART

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Citing ART

If you use ART for research, please consider citing the following reference paper:
```
@article{art2018,
    title = {Adversarial Robustness Toolbox v1.2.0},
    author = {Nicolae, Maria-Irina and Sinn, Mathieu and Tran, Minh~Ngoc and Buesser, Beat and Rawat, Ambrish and Wistuba, Martin and Zantedeschi, Valentina and Baracaldo, Nathalie and Chen, Bryant and Ludwig, Heiko and Molloy, Ian and Edwards, Ben},
    journal = {CoRR},
    volume = {1807.01069},
    year = {2018},
    url = {https://arxiv.org/pdf/1807.01069}
}
```
