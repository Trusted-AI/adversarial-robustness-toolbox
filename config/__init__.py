from __future__ import absolute_import, division, print_function

from os.path import abspath, dirname, expanduser, join
import sys

# Ensure compatibility with Python 2 and 3 when using ConfigParser
if sys.version_info >= (3, 0):
    import configparser as cp
else:
    import ConfigParser as cp


def config_to_dict(section):
    dict = {}
    options = parser.options(section)
    for option in options:
        try:
            dict[option] = parser.get(section, option)
        except:
            print("exception on %s!" % option)
            dict[option] = None
    return dict


__all__ = ['config_dict', 'MNIST_PATH', 'CIFAR10_PATH', 'IMAGENET_PATH', 'STL10_PATH', 'DATA_PATH']

config_file = join(dirname(__file__), 'config.ini')
parser = cp.ConfigParser()
if sys.version_info >= (3, 0):
    with open(config_file, mode='rt') as f:
        parser.read_file(f)
else:
    parser.read(config_file)

# Generate config dictionary
config_dict = dict()
if sys.version_info >= (3, 0):
    config_dict.update(parser['DEFAULT'])
else:
    config_dict['profile'] = parser.get('DEFAULT', 'profile')

profile = config_dict['profile'] = config_dict.get('profile')

# Load the configuration for the current profile
if parser.has_section(profile):
    if sys.version_info >= (3, 0):
        config_dict.update(parser[profile])
    else:
        config_dict.update(config_to_dict(profile))

# Add configured paths to PYTHONPATH
for key in config_dict:
    path = abspath(expanduser(config_dict[key]))
    sys.path.append(path)

MNIST_PATH = abspath(expanduser(config_dict['mnist_path']))
CIFAR10_PATH = abspath(expanduser(config_dict['cifar10_path']))
IMAGENET_PATH = abspath(expanduser(config_dict['imagenet_path']))
STL10_PATH = abspath(expanduser(config_dict['stl10_path']))
DATA_PATH = abspath(expanduser(config_dict['data_path']))
