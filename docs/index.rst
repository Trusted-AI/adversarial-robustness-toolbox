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

Supported Attack and Defense Methods
------------------------------------

The Adversarial Robustness Toolbox contains implementations of the following attacks:

* Deep Fool (`Moosavi-Dezfooli et al., 2015a`_)
* Fast Gradient Method (`Goodfellow et al., 2014`_)
* Jacobian Saliency Map (`Papernot et al., 2016`_)
* Universal Perturbation (`Moosavi-Dezfooli et al., 2016`_)
* Virtual Adversarial Method (`Moosavi-Dezfooli et al., 2015b`_)
* C&W Attack (`Carlini and Wagner, 2016`_)
* NewtonFool (`Jang et al., 2017`_)

The following defense methods are also supported:

* Feature squeezing (`Xu et al., 2017`_)
* Spatial smoothing (`Xu et al., 2017`_)
* Label smoothing (`Warde-Farley and Goodfellow, 2016`_)
* Adversarial training (`Szegedy et al., 2013`_)
* Virtual adversarial training (`Miyato et al., 2017`_)


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
   modules/defences
   modules/detection
   modules/metrics
   modules/utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https:github.com/IBM/adversarial-robustness-toolbox
.. _Moosavi-Dezfooli et al., 2015a: https://arxiv.org/abs/1511.04599
.. _Goodfellow et al., 2014: https://arxiv.org/abs/1412.6572
.. _Papernot et al., 2016: https://arxiv.org/abs/1511.07528
.. _Moosavi-Dezfooli et al., 2016: https://arxiv.org/abs/1610.08401
.. _Moosavi-Dezfooli et al., 2015b: https://arxiv.org/abs/1507.00677
.. _Carlini and Wagner, 2016: https://arxiv.org/abs/1608.04644
.. _Jang et al., 2017: http://doi.acm.org/10.1145/3134600.3134635
.. _Xu et al., 2017: http://arxiv.org/abs/1704.01155
.. _Warde-Farley and Goodfellow, 2016: https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf
.. _Szegedy et al., 2013: http://arxiv.org/abs/1312.6199
.. _Miyato et al., 2017: https://arxiv.org/abs/1704.03976
