Setup
=====

The Adversarial Robustness Toolbox is designed to run with Python 3 and Python 2 (with small changes).
You can either download the source code or clone the repository in your directory of choice:

.. code-block:: bash

   git clone https://github.com/IBM/adversarial-robustness-toolbox

To install the project dependencies, use the requirements file:

.. code-block:: bash

   pip install .

The library comes with a basic set of unit tests.
To check your install, you can run all the unit tests by calling in the library folder:

.. code-block:: bash

   bash run_tests.sh

The configuration file `config/config.ini` allows to set custom paths for data.
By default, data is downloaded in the `data` folder as follows:

.. code-block:: none

   [DEFAULT]
   profile=LOCAL

   [LOCAL]
   data_path=./data
   mnist_path=./data/mnist
   cifar10_path=./data/cifar-10
   stl10_path=./data/stl-10

If the datasets are not present at the indicated path, loading them will also download the data.
