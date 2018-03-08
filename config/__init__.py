from __future__ import absolute_import, division, print_function

from os.path import abspath, dirname, expanduser, join
import sys

import configparser

__all__ = ['config_dict', 'MNIST_PATH', 'CIFAR10_PATH', 'IMAGENET_PATH', 'STL10_PATH', 'DATA_PATH']

config_file = join(dirname(__file__), 'config.ini')
parser = configparser.ConfigParser()
with open(config_file, mode='rt') as f:
    parser.read_file(f)

# Generate config dictionary
config_dict = dict()
config_dict.update(parser['DEFAULT'])

profile = config_dict['profile'] = config_dict.get('profile', 'profile1')

# Load the configuration for the current profile
if parser.has_section(profile):
    config_dict.update(parser[profile])

# Add configured paths to PYTHONPATH
for key in config_dict:
    path = abspath(expanduser(config_dict[key]))
    sys.path.append(path)

MNIST_PATH = abspath(expanduser(config_dict['mnist_path']))
CIFAR10_PATH = abspath(expanduser(config_dict['cifar10_path']))
IMAGENET_PATH = abspath(expanduser(config_dict['imagenet_path']))
STL10_PATH = abspath(expanduser(config_dict['stl10_path']))
DATA_PATH = abspath(expanduser(config_dict['data_path']))
