import json
import logging
import logging.config
import os

from numpy import float32

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'std': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M'
        }
    },
    'handlers': {
        'default': {
            'class': 'logging.NullHandler',
        },
        'test': {
            'class': 'logging.StreamHandler',
            'formatter': 'std',
            'level': logging.DEBUG
        }
    },
    'loggers': {
        '': {
            'handlers': ['default']
        },
        'testLogger': {
            'handlers': ['test'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

_folder = os.path.expanduser('~')
if not os.access(_folder, os.W_OK):
    _folder = '/tmp'
_folder = os.path.join(_folder, '.art')

_config_path = os.path.expanduser(os.path.join(_folder, 'config.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:
        logger.warning('Unable to create folder for configuration file.', exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {'DATA_PATH': os.path.join(_folder, 'data')}

    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        logger.warning('Unable to create configuration file', exc_info=True)

if 'DATA_PATH' in _config:
    DATA_PATH = _config['DATA_PATH']

NUMPY_DTYPE = float32
