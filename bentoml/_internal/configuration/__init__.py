import os
import re
import typing as t
import logging
from functools import lru_cache

import yaml

from bentoml.exceptions import BentoMLException

from ..utils.pkg import get_pkg_version
from ..utils.pkg import source_locations

# Note this file is loaded prior to logging being configured, thus logger is only
# used within functions in this file
logger = logging.getLogger(__name__)

DEBUG_ENV_VAR = "BENTOML_DEBUG"
QUIET_ENV_VAR = "BENTOML_QUIET"
CONFIG_ENV_VAR = "BENTOML_CONFIG"


def expand_env_var(env_var: str) -> str:
    """Expands potentially nested env var by repeatedly applying `expandvars` and
    `expanduser` until interpolation stops having any effect.
    """
    while True:
        interpolated = os.path.expanduser(os.path.expandvars(str(env_var)))
        if interpolated == env_var:
            return interpolated
        else:
            env_var = interpolated


def clean_bentoml_version(bentoml_version: str) -> str:
    post_version = bentoml_version.split("+")[0]
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:(a|rc)\d)*", post_version)
    if match is None:
        raise BentoMLException("Errors while parsing BentoML version.")
    return match.group()


# Find BentoML version managed by setuptools_scm
BENTOML_VERSION: str = get_pkg_version("bentoml")
# Get clean BentoML version indicating latest PyPI release. E.g. 1.0.0.post => 1.0.0
CLEAN_BENTOML_VERSION: str = clean_bentoml_version(BENTOML_VERSION)


@lru_cache(maxsize=1)
def is_pypi_installed_bentoml() -> bool:
    # We can use importlib.util.find_spec("bentoml").submodule_search_locations
    # to find the source code of bentoml. If bentoml is installed with pypi, then it should
    # be installed in 'site-packages', otherwise, it will return path to source git.
    bentoml_path = source_locations("bentoml")
    if bentoml_path and "site-packages" in bentoml_path:
        return True
    return False


def get_bentoml_config_file_from_env() -> t.Optional[str]:
    if CONFIG_ENV_VAR in os.environ:
        # User local config file for customizing bentoml
        return expand_env_var(os.environ.get(CONFIG_ENV_VAR, ""))
    return None


def set_debug_mode(enabled: bool) -> None:
    os.environ[DEBUG_ENV_VAR] = str(enabled)

    logger.info(
        f"{'Enabling' if enabled else 'Disabling'} debug mode for current BentoML session"
    )


def get_debug_mode() -> bool:
    if DEBUG_ENV_VAR in os.environ:
        return os.environ[DEBUG_ENV_VAR].lower() == "true"
    return False


def set_quiet_mode(enabled: bool) -> None:
    os.environ[DEBUG_ENV_VAR] = str(enabled)

    # do not log setting quiet mode


def get_quiet_mode() -> bool:
    if QUIET_ENV_VAR in os.environ:
        return os.environ[QUIET_ENV_VAR].lower() == "true"
    return False


def load_global_config(bentoml_config_file: t.Optional[str] = None):
    """Load global configuration of BentoML"""

    from .containers import BentoMLContainer
    from .containers import BentoMLConfiguration

    if not bentoml_config_file:
        bentoml_config_file = get_bentoml_config_file_from_env()

    if bentoml_config_file:
        if not bentoml_config_file.endswith((".yml", ".yaml")):
            raise Exception(
                "BentoML config file specified in ENV VAR does not end with `.yaml`: "
                f"`BENTOML_CONFIG={bentoml_config_file}`"
            )
        if not os.path.isfile(bentoml_config_file):
            raise FileNotFoundError(
                "BentoML config file specified in ENV VAR not found: "
                f"`BENTOML_CONFIG={bentoml_config_file}`"
            )

    bentoml_configuration = BentoMLConfiguration(
        override_config_file=bentoml_config_file,
    )

    BentoMLContainer.config.set(bentoml_configuration.as_dict())


def save_global_config(config_file_handle: t.IO[t.Any]):
    from ..configuration.containers import BentoMLContainer

    content = yaml.safe_dump(BentoMLContainer.config)
    config_file_handle.write(content)
