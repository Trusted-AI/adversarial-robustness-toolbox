# Adversarial Robustness Toolbox (ART) v1.14
<p align="center">
  <img src="docs/images/art_lfai.png?raw=true" width="467" title="ART logo">
</p>
<br />

![Continuous Integration](https://github.com/Trusted-AI/adversarial-robustness-toolbox/workflows/Continuous%20Integration/badge.svg)
![CodeQL](https://github.com/Trusted-AI/adversarial-robustness-toolbox/workflows/CodeQL/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/adversarial-robustness-toolbox/badge/?version=latest)](http://adversarial-robustness-toolbox.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://badge.fury.io/py/adversarial-robustness-toolbox.svg)](https://badge.fury.io/py/adversarial-robustness-toolbox)
[![codecov](https://codecov.io/gh/Trusted-AI/adversarial-robustness-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/Trusted-AI/adversarial-robustness-toolbox)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adversarial-robustness-toolbox)](https://pypi.org/project/adversarial-robustness-toolbox/)
[![slack-img](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://ibm-art.slack.com/)
[![Downloads](https://pepy.tech/badge/adversarial-robustness-toolbox)](https://pepy.tech/project/adversarial-robustness-toolbox)
[![Downloads](https://pepy.tech/badge/adversarial-robustness-toolbox/month)](https://pepy.tech/project/adversarial-robustness-toolbox)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5090/badge)](https://bestpractices.coreinfrastructure.org/projects/5090)

<p align="center">
  <img src="https://raw.githubusercontent.com/lfai/artwork/master/lfaidata-assets/lfaidata-project-badge/graduate/color/lfaidata-project-badge-graduate-color.png" alt="LF AI & Data" width="300"/>
</p>

对抗性鲁棒性工具集（ART）是用于机器学习安全性的Python库。ART 由
[Linux Foundation AI & Data Foundation](https://lfaidata.foundation) (LF AI & Data)。 ART提供的工具可
帮助开发人员和研究人员针对以下方面捍卫和评估机器学习模型和应用程序：
逃逸，数据污染，模型提取和推断的对抗性威胁。ART支持所有流行的机器学习框架
（TensorFlow，Keras，PyTorch，MXNet，scikit-learn，XGBoost，LightGBM，CatBoost，GPy等），所有数据类型
（图像，表格，音频，视频等）和机器学习任务（分类，物体检测，语音识别，
生成模型，认证等）。

## Adversarial Threats

<p align="center">
  <img src="docs/images/adversarial_threats_attacker.png?raw=true" width="400" title="ART logo">
  <img src="docs/images/adversarial_threats_art.png?raw=true" width="400" title="ART logo">
</p>
<br />

## ART for Red and Blue Teams (selection)

<p align="center">
  <img src="docs/images/white_hat_blue_red.png?raw=true" width="800" title="ART Red and Blue Teams">
</p>
<br />

## 学到更多

| **[开始使用][get-started]**     | **[文献资料][documentation]**     | **[贡献][contributing]**           |
|-------------------------------------|-------------------------------|-----------------------------------|
| - [安装][installation]<br>- [例子](examples/README.md)<br>- [Notebooks](notebooks/README.md) | - [攻击][attacks]<br>- [防御][defences]<br>- [评估器][estimators]<br>- [指标][metrics]<br>- [技术文档](https://adversarial-robustness-toolbox.readthedocs.io) | - [Slack](https://ibm-art.slack.com), [邀请函](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTA4NGQ1OTMxMzFmY2Q1MzE1NWI2MmEzN2FjNGNjOGVlODVkZDE0MjA1NTA4OGVkMjVkNmQ4MTY1NmMyOGM5YTg)<br>- [贡献](CONTRIBUTING.md)<br>- [路线图][roadmap]<br>- [引用][citing] |

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

该库正在不断开发中。欢迎反馈，错误报告和贡献！

# 致谢

本材料部分基于国防高级研究计划局（DARPA）支持的工作，合同编号HR001120C0013。
本材料中表达的任何意见，发现和结论或建议均为作者的观点，并不一定反映国防高级研究计划局（DARPA）的观点。
