import os

import json

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
        # Log warning here
        pass

if not os.path.exists(_config_path):
    # Generate default config
    _config = {'DATA_PATH': os.path.join(_folder, 'data')}

    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Log warning here
        pass

if 'DATA_PATH' in _config:
    DATA_PATH = _config['DATA_PATH']
