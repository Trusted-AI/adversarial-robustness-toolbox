# Adversarial Robustness Toolbox (ART) v1.2
<p align="center">
  <img src="docs/images/art_logo_3d_1.png?raw=true" width="200" title="ART logo">
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

Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. ART provides tools that enable
developers and researchers to evaluate, defend, certify and verify Machine Learning models and applications against the
adversarial threats of Evasion, Poisoning, Extraction, and Inference. ART supports all popular machine learning frameworks,
all data types (images, tables, audio, video, etc.) and machine learning tasks (classification, object detection,
generation, certification, etc.).

<p align="center">
  <img src="docs/images/adversarial_threats_attacker.png?raw=true" width="400" title="ART logo">
  <img src="docs/images/adversarial_threats_art.png?raw=true" width="400" title="ART logo">
</p>
<br />

Documentation of ART: https://adversarial-robustness-toolbox.readthedocs.io

Get started with [examples](examples/README.md) and [tutorials](notebooks/README.md)

The library is under continuous development. Feedback, bug reports and contributions are very welcome. 
Get in touch with us on [Slack](https://ibm-art.slack.com) (invite [here](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTA4NGQ1OTMxMzFmY2Q1MzE1NWI2MmEzN2FjNGNjOGVlODVkZDE0MjA1NTA4OGVkMjVkNmQ4MTY1NmMyOGM5YTg))!

## Supported Machine Learning Libraries
* TensorFlow (v1 and v2) (www.tensorflow.org)
* Keras (www.keras.io)
* PyTorch (www.pytorch.org)
* MXNet (https://mxnet.apache.org)
* Scikit-learn (www.scikit-learn.org)
* XGBoost (www.xgboost.ai)
* LightGBM (https://lightgbm.readthedocs.io)
* CatBoost (www.catboost.ai)
* GPy (https://sheffieldml.github.io/GPy/)

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
