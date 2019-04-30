.. adversarial-robustness-toolbox documentation master file, created by
   sphinx-quickstart on Fri Mar 23 17:02:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Adversarial Robustness Toolbox
=============================================

This is a library dedicated to **adversarial machine learning**.
Its purpose is to allow rapid crafting and analysis of attacks and defense methods for machine learning models.
The Adversarial Robustness Toolbox provides an implementation for many state-of-the-art methods for attacking and defending classifiers.
The code can be found on `GitHub`_.

The library is still under development. Feedback, bug reports and extensions are highly appreciated.

Supported Attacks, Defences and Metrics
---------------------------------------

The Adversarial Robustness Toolbox contains implementations of the following evasion attacks:

* DeepFool (`Moosavi-Dezfooli et al., 2015`_)
* Fast gradient method (`Goodfellow et al., 2014`_)
* Basic iterative method (`Kurakin et al., 2016`_)
* Projected gradient descent (`Madry et al., 2017`_)
* Jacobian saliency map (`Papernot et al., 2016`_)
* Universal perturbation (`Moosavi-Dezfooli et al., 2016`_)
* Virtual adversarial method (`Miyato et al., 2015`_)
* C&W L_2 and L_inf attacks (`Carlini and Wagner, 2016`_)
* NewtonFool (`Jang et al., 2017`_)
* Elastic net attack (`Chen et al., 2017a`_)
* Spatial transformations attack (`Engstrom et al., 2017`_)
* Query-efficient black-box attack (`Ilyas et al., 2017`_)
* Zeroth-order optimization attack (`Chen et al., 2017b`_)
* Decision-based attack (`Brendel et al., 2018`_)
* Adversarial patch (`Brown et al., 2017`_)

The following defense methods are also supported:

* Feature squeezing (`Xu et al., 2017`_)
* Spatial smoothing (`Xu et al., 2017`_)
* Label smoothing (`Warde-Farley and Goodfellow, 2016`_)
* Adversarial training (`Szegedy et al., 2013`_)
* Virtual adversarial training (`Miyato et al., 2015`_)
* Gaussian data augmentation (`Zantedeschi et al., 2017`_)
* Thermometer encoding (`Buckman et al., 2018`_)
* Total variance minimization (`Guo et al., 2018`_)
* JPEG compression (`Dziugaite et al., 2016`_)
* PixelDefend (`Song et al., 2017`_)

ART also implements detection methods of adversarial samples:

* Basic detector based on inputs
* Detector trained on the activations of a specific layer

The following detector of poisoning attacks is also supported:
* Detector based on activations analysis (`Chen et al., 2018`_)

Robustness metrics:

* CLEVER (`Weng et al., 2018`_)
* Empirical robustness (`Moosavi-Dezfooli et al., 2015`_)
* Loss sensitivity (`Arpit et al., 2017`_)


.. toctree::
   :maxdepth: 2
   :caption: User guide

   guide/setup
   guide/usage

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/attacks
   modules/classifiers
   modules/data_generators
   modules/defences
   modules/detection
   modules/poison_detection
   modules/metrics
   modules/utils
   modules/wrappers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https:github.com/IBM/adversarial-robustness-toolbox
.. _Moosavi-Dezfooli et al., 2015: https://arxiv.org/abs/1511.04599
.. _Goodfellow et al., 2014: https://arxiv.org/abs/1412.6572
.. _Kurakin et al., 2016: https://arxiv.org/abs/1607.02533
.. _Madry et al., 2017: https://arxiv.org/abs/1706.06083
.. _Papernot et al., 2016: https://arxiv.org/abs/1511.07528
.. _Moosavi-Dezfooli et al., 2016: https://arxiv.org/abs/1610.08401
.. _Carlini and Wagner, 2016: https://arxiv.org/abs/1608.04644
.. _Jang et al., 2017: http://doi.acm.org/10.1145/3134600.3134635
.. _Chen et al., 2017a: https://arxiv.org/abs/1709.04114
.. _Chen et al., 2017b: https://arxiv.org/abs/1708.03999
.. _Engstrom et al., 2017: https://arxiv.org/abs/1712.02779
.. _Ilyas et al., 2017: https://arxiv.org/abs/1712.07113
.. _Xu et al., 2017: http://arxiv.org/abs/1704.01155
.. _Warde-Farley and Goodfellow, 2016: https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf
.. _Szegedy et al., 2013: http://arxiv.org/abs/1312.6199
.. _Miyato et al., 2015: https://arxiv.org/abs/1507.00677
.. _Zantedeschi et al., 2017: https://arxiv.org/abs/1707.06728
.. _Buckman et al., 2018: https://openreview.net/forum?id=S18Su--CW
.. _Guo et al., 2018: https://openreview.net/forum?id=SyJ7ClWCb
.. _Dziugaite et al., 2016: https://arxiv.org/abs/1608.00853
.. _Song et al., 2017: https://arxiv.org/abs/1710.10766
.. _Chen et al., 2018: https://arxiv.org/abs/1811.03728
.. _Weng et al., 2018: https://arxiv.org/abs/1801.10578
.. _Arpit et al., 2017: https://arxiv.org/abs/1706.05394
.. _Brendel et al., 2018: https://arxiv.org/abs/1712.04248
.. _Brown et al., 2017: https://arxiv.org/abs/1712.09665
