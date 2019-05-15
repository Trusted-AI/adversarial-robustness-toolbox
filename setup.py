from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy',
                    'scipy',
                    'six',
                    'setuptools',
                    'scikit-learn']

tests_require = ['mxnet',
                 'h5py',
                 'keras',
                 'Pillow',
                 'requests',
                 'tensorflow',
                 'torch == 0.4.0']

docs_require = ['sphinx >= 1.4',
                'sphinx_rtd_theme']

setup(name='Adversarial Robustness Toolbox',
      version='0.8.0',
      description='IBM Adversarial machine learning toolbox',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Irina Nicolae',
      author_email='maria-irina.nicolae@ibm.com',
      url='https://github.com/IBM/adversarial-robustness-toolbox',
      license='MIT',
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require={
          'tests': tests_require,
          'docs': docs_require
      },
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
      packages=find_packages(),
      include_package_data=True)
