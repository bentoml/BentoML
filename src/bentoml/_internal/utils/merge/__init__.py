from __future__ import annotations

import typing as t


def deep_merge(base: dict[str, t.Any], update: dict[str, t.Any]) -> dict[str, t.Any]:
    """Recursively merge two dictionaries, with values from update taking precedence.

    This function implements the same merge behavior as the deepmerge package:
    - Recursively merge dictionaries
    - Override all other types with values from update
    - Override conflicting types with values from update

    Args:
        base: The base dictionary
        update: The dictionary to merge on top of base

    Returns:
        The merged dictionary
    """
    result = base.copy()
    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
