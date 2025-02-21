from __future__ import annotations

import typing as t
from typing import Any
from typing import Dict
from typing import TypeVar

T = TypeVar("T", bound=Dict[str, Any])
MergeDict = Dict[str, Any]


def deep_merge(base: T, update: Dict[str, Any]) -> T:
    """Recursively merge two dictionaries, with values from update taking precedence.

    This function implements the same merge behavior as the deepmerge package:
    - Recursively merge dictionaries
    - Override all other types with values from update
    - Override conflicting types with values from update

    Args:
        base: The base dictionary
        update: The dictionary to merge on top of base

    Returns:
        The merged dictionary with the same type as base
    """
    result = t.cast(T, base.copy())
    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(
                t.cast(T, result[key]), t.cast(Dict[str, Any], value)
            )
        else:
            result[key] = value
    return result
