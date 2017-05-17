import os
import sys

import configparser

__all__ = ['config_dict', 'MNIST_PATH', 'CIFAR10_PATH', 'DATA_PATH']

config_file = os.path.join(os.path.dirname(__file__), 'config.ini')
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
    path = os.path.abspath(os.path.expanduser(config_dict[key]))
    sys.path.append(path)

MNIST_PATH = os.path.abspath(os.path.expanduser(config_dict['mnist_path']))
CIFAR10_PATH = os.path.abspath(os.path.expanduser(config_dict['cifar10_path']))
DATA_PATH = os.path.abspath(os.path.expanduser(config_dict['data_path']))