# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from pathlib import Path

from bentoml import __version__
from bentoml.utils import _is_pypi_release
from bentoml.exceptions import BentoMLConfigException
from bentoml.configuration.configparser import BentoMLConfigParser


# Note this file is loaded prior to logging being configured, thus logger is only
# used within functions in this file
logger = logging.getLogger(__name__)


# Default bentoml config comes with the library bentoml/config/default_bentoml.cfg
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "default_bentoml.cfg")


def expand_env_var(env_var):
    """Expands potentially nested env var by repeatedly applying `expandvars` and
    `expanduser` until interpolation stops having any effect.
    """
    if not env_var:
        return env_var
    while True:
        interpolated = os.path.expanduser(os.path.expandvars(str(env_var)))
        if interpolated == env_var:
            return interpolated
        else:
            env_var = interpolated


def parameterized_config(template):
    """Generates a configuration from the provided template + variables defined in
    current scope

    Args:
        :param template: a config content templated with {{variables}}
    Returns:
        string: config content after templated with locals() and globals()
    """
    all_vars = {k: v for d in [globals(), locals()] for k, v in d.items()}
    return template.format(**all_vars)


DEFAULT_BENTOML_HOME = expand_env_var(os.environ.get("BENTOML_HOME", "~/bentoml"))
BENTOML_HOME = DEFAULT_BENTOML_HOME

# This is used as default for config('core', 'bentoml_deploy_version') - which is used
# for getting the BentoML PyPI version string or the URL to a BentoML sdist, indicating
# the BentoML module to be used when loading and using a saved BentoService bundle.
# This is useful when using customized BentoML fork/branch or when working with
# development branches of BentoML
BENTOML_VERSION = __version__

# e.g. from '0.4.2+5.g6cac97f.dirty' to '0.4.2'
PREV_PYPI_RELEASE_VERSION = __version__.split('+')[0]


if not _is_pypi_release():
    # Reset to LAST_PYPI_RELEASE_VERSION if bentoml module is 'dirty'
    # This will be used as default value of 'core/deploy_bentoml_version' config
    BENTOML_VERSION = PREV_PYPI_RELEASE_VERSION


def get_local_config_file():
    if "BENTOML_CONFIG" in os.environ:
        # User local config file for customizing bentoml
        return expand_env_var(os.environ.get("BENTOML_CONFIG"))
    else:
        return os.path.join(BENTOML_HOME, "bentoml.cfg")


def load_config():
    global BENTOML_HOME  # pylint: disable=global-statement

    try:
        Path(BENTOML_HOME).mkdir(exist_ok=True)
    except OSError as err:
        raise BentoMLConfigException(
            "Error creating bentoml home directory '{}': {}".format(
                BENTOML_HOME, err.strerror
            )
        )

    with open(DEFAULT_CONFIG_FILE, "rb") as f:
        DEFAULT_CONFIG = f.read().decode("utf-8")

    loaded_config = BentoMLConfigParser(
        default_config=parameterized_config(DEFAULT_CONFIG)
    )

    local_config_file = get_local_config_file()
    if os.path.isfile(local_config_file):
        logger.info("Loading local BentoML config file: %s", local_config_file)
        with open(local_config_file, "rb") as f:
            loaded_config.read_string(parameterized_config(f.read().decode("utf-8")))
    else:
        logger.info("No local BentoML config file found, using default configurations")

    return loaded_config


_config = None


def _reset_bentoml_home(new_bentoml_home_directory):
    global _config  # pylint: disable=global-statement
    global DEFAULT_BENTOML_HOME, BENTOML_HOME  # pylint: disable=global-statement

    DEFAULT_BENTOML_HOME = new_bentoml_home_directory
    BENTOML_HOME = new_bentoml_home_directory

    # reload config
    _config = load_config()

    # re-config logging
    from bentoml.utils.log import configure_logging

    root = logging.getLogger()
    map(root.removeHandler, root.handlers[:])
    map(root.removeFilter, root.filters[:])
    configure_logging()


def _get_bentoml_home():
    return BENTOML_HOME


def config(section=None):
    global _config  # pylint: disable=global-statement

    if _config is None:
        _config = load_config()

    if section is not None:
        return _config[section]
    else:
        return _config


def get_bentoml_deploy_version():
    """
    BentoML version to use for generated docker image or serverless function bundle to
    be deployed, this can be changed to an url to your fork of BentoML on github, or an
    url to your custom BentoML build, for example:

    bentoml_deploy_version = git+https://github.com/{username}/bentoml.git@{branch}
    """
    bentoml_deploy_version = config('core').get('bentoml_deploy_version')

    if bentoml_deploy_version != __version__:
        logger.warning(
            "BentoML local changes detected - Local BentoML repository including all "
            "code changes will be bundled together with the BentoService bundle. "
            "When used with docker, the base docker image will be default to same "
            "version as last PyPI release at version: %s. You can also force bentoml "
            "to use a specific version for deploying your BentoService bundle, "
            "by setting the config 'core/bentoml_deploy_version' to a pinned version "
            "or your custom BentoML on github, e.g.:"
            "'bentoml_deploy_version = git+https://github.com/{username}/bentoml.git@{"
            "branch}'",
            PREV_PYPI_RELEASE_VERSION,
        )
    return bentoml_deploy_version
