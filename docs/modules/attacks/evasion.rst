:mod:`art.attacks.evasion`
==========================
.. automodule:: art.attacks.evasion

Adversarial Patch
-----------------
.. autoclass:: AdversarialPatch
   :members:
   :special-members:

Adversarial Patch - Numpy
-------------------------
.. autoclass:: AdversarialPatchNumpy
   :members:
   :special-members:

Adversarial Patch - PyTorch
---------------------------
.. autoclass:: AdversarialPatchPyTorch
   :members:
   :special-members:

Adversarial Patch - TensorFlowV2
--------------------------------
.. autoclass:: AdversarialPatchTensorFlowV2
   :members:
   :special-members:

Adversarial Texture - PyTorch
-----------------------------
.. autoclass:: AdversarialTexturePyTorch
   :members:
   :special-members:

Auto Attack
-----------
.. autoclass:: AutoAttack
   :members:
   :special-members:

Auto Projected Gradient Descent (Auto-PGD)
------------------------------------------
.. autoclass:: AutoProjectedGradientDescent
   :members:
   :special-members:

Auto Conjugate Gradient (Auto-CG)
---------------------------------
.. autoclass:: AutoConjugateGradient
   :members:
   :special-members:

Rescaling-Auto Conjugate Gradient (ReACG)
---------------------------------
.. autoclass:: RescalingAutoConjugateGradient
   :members:
   :special-members:

Boundary Attack / Decision-Based Attack
---------------------------------------
.. autoclass:: BoundaryAttack
   :members:
   :special-members:

Brendel and Bethge Attack
-------------------------
.. autoclass:: BrendelBethgeAttack
   :members:
   :special-members:

Carlini and Wagner L_0 Attack
-----------------------------
.. autoclass:: CarliniL0Method
   :members:
   :special-members:

Carlini and Wagner L_2 Attack
-----------------------------
.. autoclass:: CarliniL2Method
   :members:
   :special-members:

Carlini and Wagner L_inf Attack
-------------------------------
.. autoclass:: CarliniLInfMethod
   :members:
   :special-members:

Carlini and Wagner ASR Attack
-----------------------------
.. autoclass:: CarliniWagnerASR
   :members:
   :special-members:

Composite Adversarial Attack - PyTorch
--------------------------------------
.. autoclass:: CompositeAdversarialAttackPyTorch
   :members:
   :special-members:

Decision Tree Attack
--------------------
.. autoclass:: DecisionTreeAttack
   :members:
   :special-members:

DeepFool
--------
.. autoclass:: DeepFool
   :members:
   :special-members:

DPatch
------
.. autoclass:: DPatch
   :members:
   :special-members:

RobustDPatch
------------
.. autoclass:: RobustDPatch
   :members:
   :special-members:

Elastic Net Attack
------------------
.. autoclass:: ElasticNet
   :members:
   :special-members:

Fast Gradient Method (FGM)
--------------------------
.. autoclass:: FastGradientMethod
   :members:
   :special-members:

Fast Minimum-Norm (FMN)
----------------------
FMN adapts a per-sample perturbation radius to search for small successful perturbations. This implementation
supports L2 and Linf, targeted and untargeted attacks, and starts from the original input. L0, L1, restarts and
adversarial starting points are outside this implementation's scope.

The classifier must expose logits and their matching class gradients, with one logit per class (two for binary
classification), and finite ``clip_values``. For an existing classifier and input array ``x_test``::

    from art.attacks.evasion import FastMinimumNorm

    attack = FastMinimumNorm(classifier, norm=2, max_iter=100, verbose=False)
    x_adversarial = attack.generate(x_test)

Use ``norm="inf"`` for Linf. Untargeted attacks use the model's original predictions when labels are omitted.
For a targeted attack, set ``targeted=True`` and pass target class indices or one-hot labels as ``y``.
The result contains the smallest successful examples encountered; unsuccessful searches return the original
inputs. Compare the returned predictions with the original or target labels to determine success. The method
does not certify a global minimum or guarantee that an adversarial example will be found.

See ``examples/fast_minimum_norm_digits.py`` for a CPU evaluation on real held-out handwritten digits.

Reference: Maura Pintor, Fabio Roli, Wieland Brendel, and Battista Biggio,
`Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints (NeurIPS 2021)
<https://proceedings.neurips.cc/paper_files/paper/2021/hash/a709909b1ea5c2bee24248203b1728a5-Abstract.html>`_.

.. autoclass:: FastMinimumNorm
   :members:
   :special-members:

Feature Adversaries - Numpy
---------------------------
.. autoclass:: FeatureAdversariesNumpy
   :members:
   :special-members:

Feature Adversaries - PyTorch
-----------------------------
.. autoclass:: FeatureAdversariesPyTorch
   :members:
   :special-members:

Feature Adversaries - TensorFlow
--------------------------------
.. autoclass:: FeatureAdversariesTensorFlowV2
   :members:
   :special-members:

Frame Saliency Attack
---------------------
.. autoclass:: FrameSaliencyAttack
   :members:
   :special-members:

Geometric Decision Based Attack
-------------------------------
.. autoclass:: GeoDA
   :members:
   :special-members:

GRAPHITE - Blackbox
-------------------
.. autoclass:: GRAPHITEBlackbox
   :members:
   :special-members:

GRAPHITE - Whitebox - PyTorch
-----------------------------
.. autoclass:: GRAPHITEWhiteboxPyTorch
   :members:
   :special-members:

High Confidence Low Uncertainty Attack
--------------------------------------
.. autoclass:: HighConfidenceLowUncertainty
   :members:
   :special-members:

HopSkipJump Attack
------------------
.. autoclass:: HopSkipJump
   :members:
   :special-members:

Imperceptible ASR Attack
------------------------
.. autoclass:: ImperceptibleASR
   :members:
   :special-members:

Imperceptible ASR Attack - PyTorch
----------------------------------
.. autoclass:: ImperceptibleASRPyTorch
   :members:
   :special-members:

Basic Iterative Method (BIM)
----------------------------
.. autoclass:: BasicIterativeMethod
   :members:
   :special-members:

Projected Gradient Descent (PGD)
--------------------------------
.. autoclass:: ProjectedGradientDescent
   :members:
   :special-members:

Projected Gradient Descent (PGD) - Numpy
----------------------------------------
.. autoclass:: ProjectedGradientDescentNumpy
   :members:
   :special-members:

Projected Gradient Descent (PGD) - PyTorch
------------------------------------------
.. autoclass:: ProjectedGradientDescentPyTorch
   :members:
   :special-members:

Projected Gradient Descent (PGD) - TensorFlowV2
-----------------------------------------------
.. autoclass:: ProjectedGradientDescentTensorFlowV2
   :members:
   :special-members:

LaserAttack
-----------
.. autoclass:: LaserAttack
   :members:
   :special-members:

LowProFool
----------
.. autoclass:: LowProFool
   :members:
   :special-members:

NewtonFool
----------
.. autoclass:: NewtonFool
   :members:
   :special-members:

Malware Gradient Descent - TensorFlow
-------------------------------------
.. autoclass:: MalwareGDTensorFlow
   :members:
   :special-members:

Over The Air Flickering Attack - PyTorch
----------------------------------------
.. autoclass:: OverTheAirFlickeringPyTorch
   :members:
   :special-members:

Overload Attack - PyTorch
-------------------------
.. autoclass:: OverloadPyTorch
   :members:
   :special-members:

PixelAttack
-----------
.. autoclass:: PixelAttack
   :members:
   :special-members:

ThresholdAttack
---------------
.. autoclass:: ThresholdAttack
   :members:
   :special-members:

Jacobian Saliency Map Attack (JSMA)
-----------------------------------
.. autoclass:: SaliencyMapMethod
   :members:
   :special-members:

Shadow Attack
-------------
.. autoclass:: ShadowAttack
   :members:
   :special-members:

Sign-OPT Attack
---------------
.. autoclass:: SignOPTAttack
   :members:
   :special-members:
   
Simple Black-box Adversarial Attack
-----------------------------------
.. autoclass:: SimBA
   :members:
   :special-members:

Spatial Transformations Attack
------------------------------
.. autoclass:: SpatialTransformation
   :members:
   :special-members:

Steal Now Attack Later Attack
-----------------------------
.. autoclass:: SNAL
   :members:
   :special-members:

Square Attack
-------------
.. autoclass:: SquareAttack
   :members:
   :special-members:

Targeted Universal Perturbation Attack
--------------------------------------
.. autoclass:: TargetedUniversalPerturbation
   :members:
   :special-members:

Universal Perturbation Attack
-----------------------------
.. autoclass:: UniversalPerturbation
   :members:
   :special-members:

Virtual Adversarial Method
--------------------------
.. autoclass:: VirtualAdversarialMethod
   :members:
   :special-members:

Wasserstein Attack
------------------
.. autoclass:: Wasserstein
   :members:
   :special-members:

Zeroth-Order Optimization (ZOO) Attack
--------------------------------------
.. autoclass:: ZooAttack
   :members:
   :special-members:
