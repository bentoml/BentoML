import logging
import os
import typing as t

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

import yaml

import bentoml._version as version_mod

# Note this file is loaded prior to logging being configured, thus logger is only
# used within functions in this file
logger = logging.getLogger(__name__)

DEBUG_ENV_VAR = "BENTOML_DEBUG"
CONFIG_ENV_VAR = "BENTOML_CONFIG"


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


# This is used as default for config('core', 'bentoml_deploy_version') - which is used
# for getting the BentoML PyPI version string or the URL to a BentoML sdist, indicating
# the BentoML module to be used when loading and using a saved BentoService bundle.
# This is useful when using customized BentoML fork/branch or when working with
# development branches of BentoML
BENTOML_VERSION: str = importlib_metadata.version("bentoml")


def is_pip_installed_bentoml():
    is_installed_package = hasattr(version_mod, "version_json")
    is_tagged = not BENTOML_VERSION.startswith("0+untagged")
    is_clean = not version_mod.version_tuple[-1].split(".")[-1].startswith("d")
    is_modified = BENTOML_VERSION != BENTOML_VERSION.split("+")[0]
    return is_installed_package and is_tagged and is_clean and is_modified


def get_bentoml_config_file_from_env():
    if CONFIG_ENV_VAR in os.environ:
        # User local config file for customizing bentoml
        return expand_env_var(os.environ.get(CONFIG_ENV_VAR))
    return None


def set_debug_mode(enabled: bool):
    os.environ[DEBUG_ENV_VAR] = str(enabled)

    # reconfigure logging
    from ..log import configure_logging

    configure_logging()

    logger.debug(
        f"Setting debug mode: {'ON' if enabled else 'OFF'} for current session"
    )


def get_debug_mode():
    if DEBUG_ENV_VAR in os.environ:
        return os.environ[DEBUG_ENV_VAR].lower() == "true"
    return False


def load_global_config(bentoml_config_file: t.Optional[str] = None):
    """Load global configuration of BentoML"""

    from ..configuration.containers import BentoMLConfiguration, BentoMLContainer

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


def save_global_config(config_file_handle: t.TextIO):
    from ..configuration.containers import BentoMLContainer

    content = yaml.safe_dump(BentoMLContainer.config)
    config_file_handle.write(content)
