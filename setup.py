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
                    'scikit-learn==0.22.1',
                    'Pillow==7.0.0']

# Comment out because of compatibility issues with numpy versions
# 'catboost',
tests_require = ['pytest-pep8',
                 'codecov'
                 'h5py',
                 'requests',
                 'keras>=2.2.5',
                 'mxnet',
                 'torch>=1.2.0',
                 'tensorflow>=1.14.0',
                 'scikit-learn==0.22.1',
                 'xgboost==1.0.0',
                 'lightgbm==2.3.1',
                 'GPy==1.9.9',
                 'numpy==1.18.1'
                 'scipy==1.4.1',
                 'statsmodels==0.11.0',
                 'cma==2.7.0']

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


setup(name='adversarial-robustness-toolbox',
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
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   'Topic :: Software Development :: Libraries',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      packages=find_packages(),
      include_package_data=True)
