import os
import re
import typing as t
import logging
from functools import lru_cache

from ...exceptions import BentoMLException
from ...exceptions import BentoMLConfigException

try:
    from ..._version import __version__
    from ..._version import version_tuple
except ImportError:
    # broken installation, we don't even try
    # unknown only works here because we do poor mans version compare
    __version__ = "unknown"
    version_tuple = (0, 0, "unknown")


# Note this file is loaded prior to logging being configured, thus logger is only
# used within functions in this file
logger = logging.getLogger(__name__)

DEBUG_ENV_VAR = "BENTOML_DEBUG"
QUIET_ENV_VAR = "BENTOML_QUIET"
CONFIG_ENV_VAR = "BENTOML_CONFIG"
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
GRPC_DEBUG_ENV_VAR = "GRPC_VERBOSITY"


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
BENTOML_VERSION = __version__
# Get clean BentoML version indicating latest PyPI release. E.g. 1.0.0.post => 1.0.0
CLEAN_BENTOML_VERSION = clean_bentoml_version(BENTOML_VERSION)


@lru_cache(maxsize=1)
def is_pypi_installed_bentoml() -> bool:
    """Returns true if BentoML is installed via PyPI official release or installed from
     source with a release tag, which should come with pre-built docker base image on
     dockerhub.

    BentoML uses setuptools_scm to manage its versions, it looks at three things:

    * the latest tag (with a version number)
    * the distance to this tag (e.g. number of revisions since latest tag)
    * workdir state (e.g. uncommitted changes since latest tag)

    BentoML uses setuptools_scm with `version_scheme = "post-release"` option, which
    uses roughly the following logic to render the version:

    * no distance and clean: {tag}
    * distance and clean: {tag}.post{distance}+{scm letter}{revision hash}
    * no distance and not clean: {tag}+dYYYYMMDD
    * distance and not clean: {tag}.post{distance}+{scm letter}{revision hash}.dYYYYMMDD

    This function looks at the version str and decide if BentoML installation is
    base on a recent official release.
    """
    # In a git repo with no tag, setuptools_scm generated version starts with "0.1."
    is_tagged = not BENTOML_VERSION.startswith("0.1.")
    is_clean = not version_tuple[-1].split(".")[-1].startswith("d")
    not_been_modified = BENTOML_VERSION == BENTOML_VERSION.split("+")[0]
    return is_tagged and is_clean and not_been_modified


def get_bentoml_config_file_from_env() -> t.Optional[str]:
    if CONFIG_ENV_VAR in os.environ:
        # User local config file for customizing bentoml
        return expand_env_var(os.environ.get(CONFIG_ENV_VAR, ""))
    return None


def set_debug_mode(enabled: bool) -> None:
    os.environ[DEBUG_ENV_VAR] = str(enabled)
    os.environ[GRPC_DEBUG_ENV_VAR] = "DEBUG"

    logger.info(
        f"{'Enabling' if enabled else 'Disabling'} debug mode for current BentoML session"
    )


def get_debug_mode() -> bool:
    if DEBUG_ENV_VAR in os.environ:
        return os.environ[DEBUG_ENV_VAR].lower() == "true"
    return False


def set_quiet_mode(enabled: bool) -> None:
    # do not log setting quiet mode
    os.environ[QUIET_ENV_VAR] = str(enabled)
    os.environ[GRPC_DEBUG_ENV_VAR] = "NONE"


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
            raise BentoMLConfigException(
                "BentoML config file specified in ENV VAR does not end with `.yaml`: "
                f"`BENTOML_CONFIG={bentoml_config_file}`"
            ) from None
        if not os.path.isfile(bentoml_config_file):
            raise FileNotFoundError(
                "BentoML config file specified in ENV VAR not found: "
                f"`BENTOML_CONFIG={bentoml_config_file}`"
            ) from None

    bentoml_configuration = BentoMLConfiguration(
        override_config_file=bentoml_config_file,
    )

    BentoMLContainer.config.set(bentoml_configuration.as_dict())
