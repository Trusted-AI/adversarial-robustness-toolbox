from setuptools import setup
from setuptools import find_packages


setup(name='Adversarial Robustness Toolbox',
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
                        'tensorflow',
                        'setuptools'],
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
