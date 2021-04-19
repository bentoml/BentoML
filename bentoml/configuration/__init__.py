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

import os
import logging
from dependency_injector.wiring import Provide, inject
from functools import lru_cache
from pathlib import Path

from bentoml import __version__, _version as version_mod
from bentoml.exceptions import BentoMLConfigException
from bentoml.configuration.configparser import BentoMLConfigParser

# from bentoml.configuration.containers import BentoMLContainer

# Note this file is loaded prior to logging being configured, thus logger is only
# used within functions in this file
logger = logging.getLogger(__name__)


# Default bentoml config comes with the library bentoml/config/default_bentoml.cfg
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "default_bentoml.cfg")

CONFIG_FILE_ENCODING = "utf-8"
DEBUG_ENV_VAR = "BENTOML_DEBUG"


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
LAST_PYPI_RELEASE_VERSION = __version__.split('+')[0]


def _is_pip_installed_bentoml():
    is_installed_package = hasattr(version_mod, 'version_json')
    is_tagged = not __version__.startswith('0+untagged')
    is_clean = not version_mod.get_versions()['dirty']
    return is_installed_package and is_tagged and is_clean


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
        DEFAULT_CONFIG = f.read().decode(CONFIG_FILE_ENCODING)

    loaded_config = BentoMLConfigParser(
        default_config=parameterized_config(DEFAULT_CONFIG)
    )

    local_config_file = get_local_config_file()
    if os.path.isfile(local_config_file) and not local_config_file.endswith(".yml"):
        logger.info("Loading local BentoML config file: %s", local_config_file)
        with open(local_config_file, "rb") as f:
            loaded_config.read_string(
                parameterized_config(f.read().decode(CONFIG_FILE_ENCODING))
            )
    else:
        logger.info("No local BentoML config file found, using default configurations")

    return loaded_config


_config = None


def config(section=None):
    global _config  # pylint: disable=global-statement

    if _config is None:
        _config = load_config()

    if section is not None:
        return _config[section]
    else:
        return _config


@lru_cache(maxsize=1)
def get_bentoml_deploy_version(bentoml_deploy_version: str):
    """
    BentoML version to use for generated docker image or serverless function bundle to
    be deployed, this can be changed to an url to your fork of BentoML on github, or an
    url to your custom BentoML build, for example:

    bentoml_deploy_version = git+https://github.com/{username}/bentoml.git@{branch}
    """

    if bentoml_deploy_version != LAST_PYPI_RELEASE_VERSION:
        logger.info(f"Setting BentoML deploy version to '{bentoml_deploy_version}'")

    if LAST_PYPI_RELEASE_VERSION != BENTOML_VERSION:
        if _is_pip_installed_bentoml():
            logger.warning(
                "Using BentoML not from official PyPI release. In order to find the "
                "same version of BentoML when deploying your BentoService, you must "
                "set the 'core/bentoml_deploy_version' config to a http/git location "
                "of your BentoML fork, e.g.: 'bentoml_deploy_version = "
                "git+https://github.com/{username}/bentoml.git@{branch}'"
            )
        else:
            logger.warning(
                "Using BentoML installed in `editable` model, the local BentoML "
                "repository including all code changes will be packaged together with "
                "saved bundle created, under the './bundled_pip_dependencies' "
                "directory of the saved bundle."
            )
    return bentoml_deploy_version


def set_debug_mode(enabled: bool):
    os.environ[DEBUG_ENV_VAR] = str(enabled)

    # reconfigure logging
    from bentoml.utils.log import configure_logging

    configure_logging()

    logger.debug(
        f"Setting debug mode: {'ON' if enabled else 'OFF'} for current session"
    )


def get_debug_mode():
    if DEBUG_ENV_VAR in os.environ:
        return os.environ[DEBUG_ENV_VAR].lower() == "true"
    return False


def inject_dependencies():
    """Inject dependencies and configuration to BentoML packages"""

    from timeit import default_timer as timer

    start = timer()

    logger.debug("Start dependency injection")

    from bentoml.configuration.containers import BentoMLContainer, BentoMLConfiguration

    config_file = get_local_config_file()
    if config_file.endswith(".yml"):
        configuration = BentoMLConfiguration(override_config_file=config_file)
    else:
        configuration = BentoMLConfiguration()

    container = BentoMLContainer()
    container.config.from_dict(configuration.as_dict())

    from bentoml import (
        marshal,
        server,
        tracing,
        cli,
        adapters,
        saved_bundle,
        service,
    )
    from bentoml.yatai import yatai_service

    container.wire(
        modules=[yatai_service],
        packages=[marshal, server, tracing, cli, adapters, saved_bundle, service],
    )

    end = timer()

    logger.debug("Dependency injection completed in %.3f seconds", end - start)
