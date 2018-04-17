Running scripts
===============
The library contains three main scripts for:

* training a classifier using (`train.py`)
* crafting adversarial examples on a trained model through (`generate_adversarial.py`)
* testing model accuracy on different test sets using (`test_accuracies.py`)

Detailed instructions for each script are available by typing

.. code-block:: bash

   python3 <script_name> -h

Examples
========

Some examples of how to use the toolbox when writing your own code can be found in the `examples` folder on `GitHub`_.
See `examples/README.md` for more information about what each example does. To run an example, use the following command:

.. code-block:: bash

   python3 examples/<example_name>.py

.. _GitHub: https://github.com/IBM/adversarial-robustness-toolbox
