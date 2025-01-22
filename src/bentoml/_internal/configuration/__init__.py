from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import typing as t
from functools import lru_cache
from warnings import warn

from packaging.version import Version

from ...exceptions import BentoMLConfigException
from ...exceptions import BentoMLException

if t.TYPE_CHECKING:
    from .containers import SerializationStrategy

# Note this file is loaded prior to logging being configured, thus logger is only
# used within functions in this file
logger = logging.getLogger(__name__)

DEBUG_ENV_VAR = "BENTOML_DEBUG"
QUIET_ENV_VAR = "BENTOML_QUIET"
VERBOSITY_ENV_VAR = "BENTOML_VERBOSITY"
CONFIG_ENV_VAR = "BENTOML_CONFIG"
CONFIG_OVERRIDE_ENV_VAR = "BENTOML_CONFIG_OPTIONS"
CONFIG_OVERRIDE_JSON_ENV_VAR = "BENTOML_CONFIG_OVERRIDES"
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
GRPC_DEBUG_ENV_VAR = "GRPC_VERBOSITY"
# The glibc version of python:3.11-slim image
DEFAULT_LOCK_PLATFORM = "x86_64-manylinux_2_36"


def get_bentoml_version() -> str:
    try:
        # Find BentoML version managed by setuptools_scm
        from ..._version import __version__
    except ImportError:
        # Find BentoML version managed from dist metadata
        __version__ = importlib.metadata.version("bentoml")

    return __version__


BENTOML_VERSION = get_bentoml_version()


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


@lru_cache(maxsize=1)
def clean_bentoml_version() -> str:
    try:
        version = Version(BENTOML_VERSION).base_version
        return str(version)
    except ValueError:
        raise BentoMLException("Errors while parsing BentoML version.") from None


@lru_cache(maxsize=1)
def is_editable_bentoml() -> bool:
    """Returns true if BentoML is installed in editable mode."""

    dist = importlib.metadata.distribution("bentoml")
    direct_url_file = next(
        (f for f in (dist.files or []) if str(f).endswith("direct_url.json")), None
    )
    if direct_url_file is None:
        return False
    direct_url_data = json.loads(direct_url_file.read_text())
    return direct_url_data.get("dir_info", {}).get("editable", False)


@lru_cache(maxsize=1)
def get_bentoml_requirement() -> str | None:
    """Returns the requirement string for BentoML."""
    dist = importlib.metadata.distribution("bentoml")
    direct_url_file = next(
        (f for f in (dist.files or []) if str(f).endswith("direct_url.json")), None
    )
    if direct_url_file is None:
        return f"bentoml=={BENTOML_VERSION}"
    direct_url_data = json.loads(direct_url_file.read_text())
    if direct_url_data.get("dir_info", {}).get("editable", False):
        return None
    if vcs_info := direct_url_data.get("vcs_info", None):
        # NOTE: uv 0.5.22 installs direct_url.json without commit_id key
        # This is a workaround to handle this case and should be removed in future
        commit_id = vcs_info.get("commit_id") or vcs_info.get("requested_revision")
        res = f"bentoml @ {vcs_info['vcs']}+{direct_url_data['url']}@{commit_id}"
    else:
        res = f"bentoml @ {direct_url_data['url']}"
    if subdirectory := direct_url_data.get("subdirectory", None):
        res += f"#subdirectory={subdirectory}"
    return res


def get_bentoml_config_file_from_env() -> str | None:
    if CONFIG_ENV_VAR in os.environ:
        # User local config file for customizing bentoml
        return expand_env_var(os.environ.get(CONFIG_ENV_VAR, ""))
    return None


def get_bentoml_override_config_from_env() -> str | None:
    if CONFIG_OVERRIDE_ENV_VAR in os.environ:
        # User local config options for customizing bentoml
        warn(
            f"Overriding BentoML config with {CONFIG_OVERRIDE_ENV_VAR} is deprecated "
            f"and will be removed in future release. Please use {CONFIG_OVERRIDE_JSON_ENV_VAR} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return os.environ.get(CONFIG_OVERRIDE_ENV_VAR, None)
    return None


def get_bentoml_override_config_json_from_env() -> dict[str, t.Any] | None:
    if CONFIG_OVERRIDE_JSON_ENV_VAR in os.environ:
        # User local config options for customizing bentoml
        values = os.environ.get(CONFIG_OVERRIDE_JSON_ENV_VAR, None)
        if values is not None and values != "":
            return json.loads(values)

    return None


def set_verbosity(verbosity: int) -> None:
    os.environ[VERBOSITY_ENV_VAR] = str(verbosity)
    if verbosity > 0:
        os.environ[DEBUG_ENV_VAR] = "true"
        os.environ[GRPC_DEBUG_ENV_VAR] = "DEBUG"
    elif verbosity < 0:
        os.environ[QUIET_ENV_VAR] = "true"
        os.environ[GRPC_DEBUG_ENV_VAR] = "NONE"


def set_debug_mode(enabled: bool = True) -> None:
    set_verbosity(1)


def set_quiet_mode(enabled: bool = True) -> None:
    set_verbosity(-1)


def get_debug_mode() -> bool:
    if VERBOSITY_ENV_VAR in os.environ:
        return int(os.environ[VERBOSITY_ENV_VAR]) > 0
    if DEBUG_ENV_VAR in os.environ:
        return os.environ[DEBUG_ENV_VAR].lower() == "true"
    return False


def get_quiet_mode() -> bool:
    if VERBOSITY_ENV_VAR in os.environ:
        return int(os.environ[VERBOSITY_ENV_VAR]) < 0
    if QUIET_ENV_VAR in os.environ:
        return os.environ[QUIET_ENV_VAR].lower() == "true"
    return False


def load_config(
    bentoml_config_file: str | None = None,
    override_defaults: dict[str, t.Any] | None = None,
    use_version: int = 1,
):
    """Load global configuration of BentoML"""

    from .containers import BentoMLConfiguration
    from .containers import BentoMLContainer

    if not bentoml_config_file:
        bentoml_config_file = get_bentoml_config_file_from_env()

    if bentoml_config_file:
        if not bentoml_config_file.endswith((".yml", ".yaml")):
            raise BentoMLConfigException(
                f"BentoML config file specified in ENV VAR does not end with either '.yaml' or '.yml': 'BENTOML_CONFIG={bentoml_config_file}'"
            ) from None
        if not os.path.isfile(bentoml_config_file):
            raise FileNotFoundError(
                f"BentoML config file specified in ENV VAR not found: 'BENTOML_CONFIG={bentoml_config_file}'"
            ) from None

    BentoMLContainer.config.set(
        BentoMLConfiguration(
            override_config_file=bentoml_config_file,
            override_config_values=get_bentoml_override_config_from_env(),
            override_defaults=override_defaults,
            override_config_json=get_bentoml_override_config_json_from_env(),
            use_version=use_version,
        ).to_dict()
    )


def save_config(config_file_handle: t.IO[t.Any]):
    import yaml

    from ..configuration.containers import BentoMLContainer

    content = yaml.safe_dump(BentoMLContainer.config)
    config_file_handle.write(content)


def set_serialization_strategy(serialization_strategy: SerializationStrategy):
    """
    Configure how bentoml.Service are serialized (pickled). Default via EXPORT_BENTO

    Args:
        serialization_strategy: One of ["EXPORT_BENTO", "LOCAL_BENTO", "REMOTE_BENTO"]
            EXPORT_BENTO: export and compress all Bento content as bytes
            LOCAL_BENTO: load Bento by tag in other processes, assuming available in local BentoStore
            REMOTE_BENTO: load Bento by tag in other processes, pull from Yatai/BentoCloud if not available locally
    """
    from ..configuration.containers import BentoMLContainer

    BentoMLContainer.serialization_strategy.set(serialization_strategy)
