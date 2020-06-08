import codecs
import os

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "matplotlib",
    "numpy",
    "scipy",
    "six",
    "setuptools",
    "scikit-learn==0.22.1",
    "Pillow==6.2.2",
]

tests_require = [
    "matplotlib==3.2.1",
    "numpy==1.18.1",
    "scipy==1.4.1",
    "six==1.13.0",
    "scikit-learn==0.22.1",
    "Pillow==6.0.0",
    "pytest-pep8==1.0.6",
    "typing_extensions==3.7.4.2",
    "codecov==2.0.22",
    "h5py==2.10.0",
    "requests==2.23.0",
    "statsmodels==0.11.0",
    "cma==2.7.0",
    "pydub==0.23",
    "tensorflow==2.1.0",
    "keras==2.3.1",
    "tensorflow_addons==0.8.2",
    "mxnet==1.6.0",
    "xgboost==1.0.0",
    "lightgbm==2.3.1",
    "torch==1.3.1",
    "torchvision==0.4.2",
    "catboost==0.22",
    "GPy==1.9.9",
    "pytest-mock==3.1.0",
    "resampy==0.2.2",
    "ffmpeg-python==0.2.0",
]

docs_require = ["sphinx >= 1.4", "sphinx_rtd_theme"]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="adversarial-robustness-toolbox",
    version=get_version("art/__init__.py"),
    description="Toolbox for adversarial machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Irina Nicolae",
    author_email="irinutza.n@gmail.com",
    maintainer="Beat Buesser",
    maintainer_email="beat.buesser@ie.ibm.com",
    url="https://github.com/IBM/adversarial-robustness-toolbox",
    license="MIT",
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={"tests": tests_require, "docs": docs_require},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    include_package_data=True,
)
