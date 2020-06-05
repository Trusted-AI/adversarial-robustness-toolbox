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

Adversarial Robustness Toolbox（ART）是一个Python库，支持研发人员保护机器学习模型（深度神经网络，梯度提升决策树，支持向量机，随机森林，Logistic回归，高斯过程，决策树，Scikit-learn管道，等）抵御对抗性威胁，使AI系统更安全。机器学习模型容易受到对抗性示例的影响，这些示例是经过特殊修改的输入（图像，文本，表格数据等），以通过机器学习模型达到预期的效果。 ART提供了构建和部署防御的工具， 并使用对抗性攻击对其进行测试。
防御机器学习模型主要用于验证模型的稳健性和模型强化. 所用方法包括前期处理输入，利用对抗样本增加训练数据以及利用实时检测方法来标记可能已被对手修改的输入等。 ART中实施的攻击使用目前最先进的威胁模型测试防御， 以此来制造机器学习模型的对抗性攻击。 

ART的文档： https://adversarial-robustness-toolbox.readthedocs.io

ART入门: [examples](examples/README.md) and [tutorials](notebooks/README.md)

ART正在不断发展中。 我们欢迎您的反馈，错误报告和对ART建设的任何贡献。 请您在[Slack](https://ibm-art.slack.com)上与我们联系（[邀请](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTlkMWY3MzgyZDA4ZDdiNzUzY2NhMjc5YmFhZTYzZGYwNDM4YTE1ODhhNDYyNmFlMGFjNWY4ODgyM2EwYTFjYTc) ）！

## 支持的机器学习Python库
* TensorFlow(v1和v2)(www.tensorflow.org)
* Keras (www.keras.io)
* PyTorch (www.pytorch.org)
* MXNet (https://mxnet.apache.org)
* Scikit-learn (www.scikit-learn.org)
* XGBoost (www.xgboost.ai)
* LightGBM (https://lightgbm.readthedocs.io)
* CatBoost (www.catboost.ai)
* GPy (https://sheffieldml.github.io/GPy/)

## ART中实施的攻击，防御，检测，指标，认证和验证

**逃避攻击：**
* 暗影攻击 ([Ghiasi et al., 2020](https://arxiv.org/abs/2003.08937))
* Wasserstein Attack([Wong et al., 2020](https://arxiv.org/abs/1902.07906))
* 门槛攻击 ([Vargas et al., 2019](https://arxiv.org/abs/1906.06026))
* 像素攻击 ([Vargas et al., 2019](https://arxiv.org/abs/1906.06026), [Su et al., 2019](https://ieeexplore.ieee.org/abstract/document/8601309/citations#citations))
* HopSkipJump攻击 ([Chen et al., 2019](https://arxiv.org/abs/1904.02144))
* 高可信度低不确定性对抗性例子 ([Grosse et al., 2018](https://arxiv.org/abs/1812.02606))
* Iterative frame saliency attack ([Inkawhich et al., 2018](https://arxiv.org/abs/1811.11875))
* DPatch ([Liu et al., 2018](https://arxiv.org/abs/1806.02299v4))
* 预计梯度下降 ([Madry et al., 2017](https://arxiv.org/abs/1706.06083))
* NewtonFool ([Jang et al., 2017](http://doi.acm.org/10.1145/3134600.3134635))
* 弹性网攻击 ([Chen et al., 2017](https://arxiv.org/abs/1709.04114))
* 空间变换攻击 ([Engstrom et al., 2017](https://arxiv.org/abs/1712.02779))
* 查询效率高的黑盒攻击 ([Ilyas et al., 2017](https://arxiv.org/abs/1712.07113))
* 零阶优化攻击 ([Chen et al., 2017](https://arxiv.org/abs/1708.03999))
* 基于决策的攻击 ([Brendel et al., 2018](https://arxiv.org/abs/1712.04248))
* 对抗性补丁 ([Brown et al., 2017](https://arxiv.org/abs/1712.09665))
* 决策树攻击 ([Papernot et al., 2016](https://arxiv.org/abs/1605.07277))
* Carlini＆Wagner（C＆W）L_2和L_inf攻击 ([Carlini and Wagner, 2016](https://arxiv.org/abs/1608.04644))
* 基本迭代法 ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))
* 雅可比显着性图 ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))
* 普遍扰动 ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1610.08401))
* Feature Adversaries ([Sabour et al., 2016](https://arxiv.org/abs/1511.05122))
* DeepFool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* 虚拟对抗方法 ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))
* 快速梯度法 ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))

**提取攻击:**
* 功能等效提取 ([Jagielski et al., 2019](https://arxiv.org/abs/1909.01838))
* Copycat CNN ([Correia-Silva et al., 2018](https://arxiv.org/abs/1806.05476))
* KnockoffNets ([Orekondy et al., 2018](https://arxiv.org/abs/1812.02766))

**中毒攻击**
* 对SVM的中毒攻击 ([Biggio et al., 2013](https://arxiv.org/abs/1206.6389))
* Backdoor Attack ([Gu, et. al., 2017](https://arxiv.org/abs/1708.06733))

**推理攻击:**

*模型反转*
* MIFace ([Fredrikson et al., 2015](https://dl.acm.org/doi/10.1145/2810103.2813677))

*属性推论*
* AttributeInferenceBlackBox
* AttributeInferenceWhiteBoxLifestyleDecisionTree ([Fredrikson et al., 2015](https://dl.acm.org/doi/10.1145/2810103.2813677))
* AttributeInferenceWhiteBoxDecisionTree ([Fredrikson et al., 2015](https://dl.acm.org/doi/10.1145/2810103.2813677))

**防御 - 预处理器：**
* 视频压缩
* 重采样 ([Yang et al., 2019](https://arxiv.org/abs/1809.10875))
* 温度计编码 ([Buckman et al., 2018](https://openreview.net/forum?id=S18Su--CW))
* MP3压缩 ([Carlini, N. & Wagner, D., 2018](https://arxiv.org/abs/1801.01944))
* 总方差最小化 ([Guo et al., 2018](https://openreview.net/forum?id=SyJ7ClWCb))
* PixelDefend ([Song et al., 2017](https://arxiv.org/abs/1710.10766))
* 逆甘 ([An Lin et al. 2019](https://arxiv.org/abs/1911.10291))
* 国防军 ([Samangouei et al. 2018](https://arxiv.org/abs/1805.06605))
* 高斯数据增强 ([Zantedeschi et al., 2017](https://arxiv.org/abs/1707.06728))
* 特征挤压 ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* 空间平滑 ([Xu et al., 2017](http://arxiv.org/abs/1704.01155))
* JPEG压缩 ([Dziugaite et al., 2016](https://arxiv.org/abs/1608.00853))
* 标签平滑 ([Warde-Farley and Goodfellow, 2016](https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf))
* 虚拟对抗训练 ([Miyato et al., 2015](https://arxiv.org/abs/1507.00677))

**防御 - 后处理器:**
* 反向乙状结肠 ([Lee et al., 2018](https://arxiv.org/abs/1806.00054))
* 随机噪声 ([Chandrasekaranet al., 2018](https://arxiv.org/abs/1811.02054))
* 类标签 ([Tramer et al., 2016](https://arxiv.org/abs/1609.02943), [Chandrasekaranet al., 2018](https://arxiv.org/abs/1811.02054))
* 高信心 ([Tramer et al., 2016](https://arxiv.org/abs/1609.02943))
* 四舍五入 ([Tramer et al., 2016](https://arxiv.org/abs/1609.02943))

**防御 - 培训师:**
* 对抗训练 ([Szegedy et al., 2013](http://arxiv.org/abs/1312.6199))
* 对抗训练 Madry PGD ([Madry et al., 2017](https://arxiv.org/abs/1706.06083))

**防御 - 变压器:**
* 防御蒸馏 ([Papernot et al., 2015](https://arxiv.org/abs/1511.04508))

**稳健性指标，认证和验证：**
* Clique方法稳健性验证 ([Hongge et al., 2019](https://arxiv.org/abs/1906.03849))
* 随机平滑 ([Cohen et al., 2019](https://arxiv.org/abs/1902.02918))
* CLEVER ([Weng et al., 2018](https://arxiv.org/abs/1801.10578))
* 损失敏感度 ([Arpit et al., 2017](https://arxiv.org/abs/1706.05394))
* 经验稳健性 ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))

**检测对抗样本：**
* 基于输入的基本检测器
* 用激活特定层训练的探测器
* 基于快速广义子集扫描的检测器 ([Speakman et al., 2018](https://arxiv.org/pdf/1810.08676))

**检测中毒攻击：**
* 基于激活分析的探测器 ([Chen et al., 2018](https://arxiv.org/abs/1811.03728))
* 根据数据来源进行检测 ([Baracaldo et al., 2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8473440))
* 基于光谱特征的检测 ([Tran et al., 2018](https://papers.nips.cc/paper/8024-spectral-signatures-in-backdoor-attacks.pdf))


## 建立
### 用`pip`安装
工具箱经过设计和测试，可以使用Python 3运行。可以使用`pip`从PyPi存储库安装ART：
```bash
pip install adversarial-robustness-toolbox
```

### 手动安装
可以从此存储库下载或克隆最新版本的ART：
```bash
git clone https://github.com/IBM/adversarial-robustness-toolbox
```

從項目文件夾中使用以下命令安裝ART `adversarial-robustness-toolbox`:

使用點子：
```bash
pip install .
```

使用Docker：
*構建ART docker映像： `make build`
*要進入ART docker環境，請運行： `make run-bash`
*從容器運行運行Jupyter筆記本`make run-jupyter` 然後復制並粘貼生成的網址以連接到該容器。  

####運行ART單元測試
ART提供可以在ART環境中運行的單元測試。 第一次運行測試，
  ART將下載必要的數據集，因此可能需要一段時間。
*注意：如果您希望使用Tensorflow 1環境運行單元測試，只需註釋掉其中一個中的tensorflow 2軟件包即可 `test_requirements.txt` 要么 `Dockerfile` 如所須

使用以下命令運行測試：

使用點子：
```bash
pip install -r test_requirements.txt
bash run_tests.sh
```

使用Docker：
`make run-test`

## ART入门
使用ART的示例可以在 `examples` 和 [examples/README.md](examples/README.md)提供概述和附加信息中找到， 其中包括了每个机器学习框架的最小示例。所有示例都可以使用以下命令运行：
```bash
python examples/<example_name>.py
```

更详细的示例和教程请在 `notebooks` 和 [notebooks/README.md](notebooks/README.md)中寻找。 

##為ART貢獻

看到 [CONTRIBUTING-cn.md](CONTRIBUTING-cn.md)

## 引用ART
如果您使用ART进行研究，请考虑引用以下参考文件：
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
