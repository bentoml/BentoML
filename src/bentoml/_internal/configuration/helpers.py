from __future__ import annotations

import os
import socket
import typing as t
import logging
from typing import TYPE_CHECKING
from functools import singledispatch

import yaml
import schema as s

from ..utils import LazyLoader
from ...exceptions import BentoMLConfigException

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger(__name__)

TRACING_TYPE = ["zipkin", "jaeger", "otlp", "in_memory"]


def import_configuration_spec(version: int) -> ModuleType:  # pragma: no cover
    return LazyLoader(
        f"v{version}",
        globals(),
        f"bentoml._internal.configuration.v{version}",
        exc_msg=f"Configuration version {version} does not exist.",
    )


@singledispatch
def depth(_: t.Any, _level: int = 0):  # pragma: no cover
    return _level


@depth.register(dict)
def _(d: dict[str, t.Any], level: int = 0, **kw: t.Any):
    return max(depth(v, level + 1, **kw) for v in d.values())


def rename_fields(
    d: dict[str, t.Any],
    current: str,
    replace_with: str | None = None,
    *,
    remove_only: bool = False,
):
    # We assume that the given dictionary is already flattened.
    # This function will rename the keys in the dictionary.
    # If `replace_with` is None, then the key will be removed.
    if depth(d) != 1:
        raise ValueError(
            "Given dictionary is not flattened. Use flatten_dict first."
        ) from None
    if current in d:
        if remove_only:
            logger.warning("Field '%s' is deprecated and will be removed." % current)
            d.pop(current)
        else:
            assert replace_with, "'replace_with' must be provided."
            logger.warning(
                "Field '%s' is deprecated and has been renamed to '%s'"
                % (current, replace_with)
            )
            d[replace_with] = d.pop(current)


punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^`{|}~"""


def flatten_dict(
    d: t.MutableMapping[str, t.Any],
    parent: str = "",
    sep: str = ".",
) -> t.Generator[tuple[str, t.Any], None, None]:
    """Flatten nested dictionary into a single level dictionary."""
    for k, v in d.items():
        k = f'"{k}"' if any(i in punctuation for i in k) else k
        nkey = parent + sep + k if parent else k
        if isinstance(v, t.MutableMapping):
            yield from flatten_dict(
                t.cast(t.MutableMapping[str, t.Any], v), parent=nkey, sep=sep
            )
        else:
            yield nkey, v


def load_config_file(path: str) -> dict[str, t.Any]:
    """Load configuration from given path."""
    if not os.path.exists(path):
        raise BentoMLConfigException(
            "Configuration file %s not found." % path
        ) from None
    with open(path, "rb") as f:
        config = yaml.safe_load(f)
    return config


def get_default_config(version: int) -> dict[str, t.Any]:
    config = load_config_file(
        os.path.join(
            os.path.dirname(__file__), f"v{version}", "default_configuration.yaml"
        )
    )
    mod = import_configuration_spec(version)
    assert hasattr(mod, "SCHEMA"), (
        "version %d does not have a validation schema" % version
    )
    try:
        mod.SCHEMA.validate(config)
    except s.SchemaError as e:
        raise BentoMLConfigException(
            "Default configuration for version %d does not conform to given schema:\n%s"
            % (version, e)
        ) from None
    return config


def validate_tracing_type(tracing_type: str) -> bool:
    return tracing_type in TRACING_TYPE


def validate_otlp_protocol(protocol: str) -> bool:
    return protocol in ["grpc", "http"]


def ensure_larger_than(target: int | float) -> t.Callable[[int | float], bool]:
    """Ensure that given value is (lower, inf]"""

    def v(value: int | float) -> bool:
        return value > target

    return v


ensure_larger_than_zero = ensure_larger_than(0)


def ensure_range(
    lower: int | float, upper: int | float
) -> t.Callable[[int | float], bool]:
    """Ensure that given value is within the range of [lower, upper]."""

    def v(value: int | float) -> bool:
        return lower <= value <= upper

    return v


def ensure_iterable_type(typ_: type) -> t.Callable[[t.MutableSequence[t.Any]], bool]:
    """Ensure that given mutable sequence has all elements of given types."""

    def v(value: t.MutableSequence[t.Any]) -> bool:
        return all(isinstance(i, typ_) for i in value)

    return v


def is_valid_ip_address(addr: str) -> bool:
    """Check if given string is a valid IP address."""
    try:
        _ = socket.inet_aton(addr)
        return True
    except socket.error:
        return False
