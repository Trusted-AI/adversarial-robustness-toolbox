# Adversarial Robustness Toolbox (ART) v1.20
<p align="center">
  <img src="https://raw.githubusercontent.com/Trusted-AI/adversarial-robustness-toolbox/main/docs/images/art_lfai.png" width="467" title="ART logo">
</p>
<br />

![CodeQL](https://github.com/Trusted-AI/adversarial-robustness-toolbox/workflows/CodeQL/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://badge.fury.io/py/adversarial-robustness-toolbox.svg)](https://badge.fury.io/py/adversarial-robustness-toolbox)
[![codecov](https://codecov.io/gh/Trusted-AI/adversarial-robustness-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/Trusted-AI/adversarial-robustness-toolbox)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adversarial-robustness-toolbox)](https://pypi.org/project/adversarial-robustness-toolbox/)
[![slack-img](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://ibm-art.slack.com/)
[![Downloads](https://static.pepy.tech/badge/adversarial-robustness-toolbox)](https://pepy.tech/project/adversarial-robustness-toolbox)
[![Downloads](https://static.pepy.tech/badge/adversarial-robustness-toolbox/month)](https://pepy.tech/project/adversarial-robustness-toolbox)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5090/badge)](https://bestpractices.coreinfrastructure.org/projects/5090)

[中文README请按此处](README-cn.md)

 <div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/lfaidata-project-badge-graduate-color_dark.png" width="400" title="LF AI & Data">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/lfaidata-project-badge-graduate-color.png" width="400" title="LF AI & Data">
    <img alt="Fallback image description" src="default-image.png" width="400">
  </picture>
</div>
<br />

Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. ART is hosted by the 
[Linux Foundation AI & Data Foundation](https://lfaidata.foundation) (LF AI & Data). ART provides tools that enable
developers and researchers to defend and evaluate Machine Learning models and applications against the
adversarial threats of Evasion, Poisoning, Extraction, and Inference. ART supports all popular machine learning frameworks
(TensorFlow, Keras, PyTorch, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.), all data types
(images, tables, audio, video, etc.) and machine learning tasks (classification, object detection, speech recognition,
generation, certification, etc.).

## Adversarial Threats

 <div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/adversarial_threats_attacker_dark.png" width="400 title="ART Threats">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/adversarial_threats_attacker.png" width="400 title="ART Threats">
    <img alt="Fallback image description" src="default-image.png" width="400">
  </picture>
</div>

<p align="center">
  <img src="docs/images/adversarial_threats_art.png?raw=true" width="400" title="ART Matrix">
</p>
<br />

## ART for Red and Blue Teams (selection)

 <div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/white_hat_blue_red_dark.png" width="800 title="ART Red and Blue Teams">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/white_hat_blue_red.png" width="800 title="ART Red and Blue Teams">
    <img alt="Fallback image description" src="default-image.png" width="800">
  </picture>
</div>
<br />

## Learn more

| **[Get Started][get-started]**     | **[Documentation][documentation]**     | **[Contributing][contributing]**           |
|-------------------------------------|-------------------------------|-----------------------------------|
| - [Installation][installation]<br>- [Examples](examples/README.md)<br>- [Notebooks](notebooks/README.md) | - [Attacks][attacks]<br>- [Defences][defences]<br>- [Estimators][estimators]<br>- [Metrics][metrics]<br>- [Technical Documentation](https://adversarial-robustness-toolbox.readthedocs.io) | - [Slack](https://ibm-art.slack.com), [Invitation](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTA4NGQ1OTMxMzFmY2Q1MzE1NWI2MmEzN2FjNGNjOGVlODVkZDE0MjA1NTA4OGVkMjVkNmQ4MTY1NmMyOGM5YTg)<br>- [Contributing](CONTRIBUTING.md)<br>- [Roadmap][roadmap]<br>- [Citing][citing] |

[get-started]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Get-Started
[attacks]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Attacks
[defences]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Defences
[estimators]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Estimators
[metrics]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Metrics
[contributing]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Contributing
[documentation]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Documentation
[installation]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Get-Started#setup
[roadmap]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Roadmap
[citing]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Contributing#citing-art

The library is under continuous development. Feedback, bug reports and contributions are very welcome!

# Acknowledgment
This material is partially based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under
Contract No. HR001120C0013. Any opinions, findings and conclusions or recommendations expressed in this material are
those of the author(s) and do not necessarily reflect the views of the Defense Advanced Research Projects Agency (DARPA).
