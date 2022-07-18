from __future__ import annotations

from types import ModuleType
from typing import cast

try:
    import importlib.metadata as importlib_metadata
    from importlib.metadata import PackageNotFoundError
except ImportError:
    import importlib_metadata
    from importlib_metadata import PackageNotFoundError

import importlib.util

from packaging.version import Version

__all__ = [
    "PackageNotFoundError",
    "pkg_version_info",
    "get_pkg_version",
    "source_locations",
    "find_spec",
]

get_pkg_version = importlib_metadata.version
find_spec = importlib.util.find_spec


def pkg_version_info(pkg_name: str | ModuleType) -> tuple[int, int, int]:
    if isinstance(pkg_name, ModuleType):
        pkg_name = pkg_name.__name__
    pkg_version = Version(get_pkg_version(pkg_name))
    return pkg_version.major, pkg_version.minor, pkg_version.micro


def source_locations(pkg: str) -> str | None:
    module = find_spec(pkg)
    if module is None:
        return
    (module_path,) = module.submodule_search_locations  # type: ignore (unfinished typed)
    return cast(str, module_path)
