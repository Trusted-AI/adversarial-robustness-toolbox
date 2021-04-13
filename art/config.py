# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module loads and provides configuration parameters for ART.
"""
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES

ART_NUMPY_DTYPE = np.float32  # pylint: disable=C0103
ART_DATA_PATH: str

# --------------------------------------------------------------------------------------------- DEFAULT PACKAGE CONFIGS

_folder = os.path.expanduser("~")
if not os.access(_folder, os.W_OK):
    _folder = "/tmp"  # pylint: disable=C0103
_folder = os.path.join(_folder, ".art")


def set_data_path(path):
    """
    Set the path for ART's data directory (ART_DATA_PATH).
    """
    expanded_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(expanded_path, exist_ok=True)
    if not os.access(expanded_path, os.R_OK):
        raise OSError(f"path {expanded_path} cannot be read from")
    if not os.access(expanded_path, os.W_OK):
        logger.warning("path %s is read only", expanded_path)

    global ART_DATA_PATH  # pylint: disable=W0603
    ART_DATA_PATH = expanded_path
    logger.info("set ART_DATA_PATH to %s", expanded_path)


# Load data from configuration file if it exists. Otherwise create one.
_config_path = os.path.expanduser(os.path.join(_folder, "config.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)

            # Since renaming this variable we must update existing config files
            if "DATA_PATH" in _config:
                _config["ART_DATA_PATH"] = _config.pop("DATA_PATH")
                try:
                    with open(_config_path, "w") as f:
                        f.write(json.dumps(_config, indent=4))
                except IOError:
                    logger.warning("Unable to update configuration file", exc_info=True)

    except ValueError:
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:
        logger.warning("Unable to create folder for configuration file.", exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {"ART_DATA_PATH": os.path.join(_folder, "data")}

    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        logger.warning("Unable to create configuration file", exc_info=True)

if "ART_DATA_PATH" in _config:
    set_data_path(_config["ART_DATA_PATH"])
