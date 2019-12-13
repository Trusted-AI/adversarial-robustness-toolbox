import codecs
import os
from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['matplotlib',
                    'numpy',
                    'scipy',
                    'six',
                    'setuptools',
                    'scikit-learn']

# Comment out because of compatibility issues with numpy versions
# 'catboost',
tests_require = ['pytest-pep8',
                 'keras>=2.2.5',
                 'h5py',
                 'mxnet',
                 'Pillow',
                 'requests',
                 'torch>=1.2.0',
                 'tensorflow>=1.13.2',
                 'scikit-learn',
                 'xgboost',
                 'lightgbm',
                 'GPy',
                 'SciPy',
                 'statsmodels']

docs_require = ['sphinx >= 1.4',
                'sphinx_rtd_theme']


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(name='adversarial_robustness_toolbox',
      version=get_version("art/__init__.py"),
      description='Toolbox for adversarial machine learning.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Irina Nicolae',
      author_email='irinutza.n@gmail.com',
      maintainer='Beat Buesser',
      maintainer_email='beat.buesser@ie.ibm.com',
      url='https://github.com/IBM/adversarial-robustness-toolbox',
      license='MIT',
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={
          'tests': tests_require,
          'docs': docs_require
      },
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3',
                   'Topic :: Software Development :: Libraries',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      packages=find_packages(),
      include_package_data=True)
