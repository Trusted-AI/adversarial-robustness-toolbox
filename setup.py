# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from setuptools import setup
from setuptools import find_packages


setup(name='adversarial_robustness_toolbox',
      version='0.1',
      description='IBM Adversarial machine learning toolbox',
      author='Irina Nicolae',
      author_email='maria-irina.nicolae@ibm.com',
      url='https://github.com/IBM/adversarial-robustness-toolbox',
      license='MIT',
      install_requires=['h5py',
                        'Keras',
                        'scipy',
                        'matplotlib',
                        'setuptools'],
      extras_require={
        "tf": ["tensorflow<=1.5.0"],
        "tf_gpu": ["tensorflow-gpu<=1.5.0"],
      },
      # extras_require={
          # 'tests': ['pytest',
          #           'pytest-pep8',
          #           'pytest-xdist',
          #           'pytest-cov'],
      # },
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      packages=find_packages())
