from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from click import Context
    from click import Parameter


def to_valid_docker_image_name(name: str) -> str:
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return name.lower().strip("._-")


def to_valid_docker_image_version(version: str) -> str:
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return version.encode("ascii", errors="ignore").decode().lstrip(".-")[:128]


def _validate_docker_tag(tag: str) -> str:
    if ":" in tag:
        name, version = tag.split(":")[:2]
    else:
        name, version = tag, None

    valid_name_pattern = re.compile(
        r"""
        ^(
        [a-z0-9]+      # alphanumeric
        (.|_{1,2}|-+)? # seperators
        )*$
        """,
        re.VERBOSE,
    )
    valid_version_pattern = re.compile(
        r"""
        ^
        [a-zA-Z0-9] # cant start with .-
        [ -~]{,127} # ascii match rest, cap at 128
        $
        """,
        re.VERBOSE,
    )

    if not valid_name_pattern.match(name):
        raise BentoMLException(
            f"Provided Docker Image tag {tag} is invalid. "
            "Name components may contain lowercase letters, digits "
            "and separators. A separator is defined as a period, "
            "one or two underscores, or one or more dashes."
        )
    if version and not valid_version_pattern.match(version):
        raise BentoMLException(
            f"Provided Docker Image tag {tag} is invalid. "
            "A tag name must be valid ASCII and may contain "
            "lowercase and uppercase letters, digits, underscores, "
            "periods and dashes. A tag name may not start with a period "
            "or a dash and may contain a maximum of 128 characters."
        )
    return tag


def validate_tag(
    ctx: Context, param: Parameter, tag: str | tuple[str] | None
) -> str | tuple[str] | None:
    if tag is None:
        return tag
    elif isinstance(tag, tuple):
        return tuple(map(_validate_docker_tag, tag))
    else:
        return _validate_docker_tag(tag)
