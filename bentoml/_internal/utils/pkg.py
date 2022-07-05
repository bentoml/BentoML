from __future__ import annotations

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

get_pkg_version = importlib_metadata.version


# mimic the behaviour of version_info assuming the pkg follows semver
def pkg_version_info(pkg_name: str) -> tuple[int, int, int]:
    return tuple(map(int, get_pkg_version(pkg_name).split(".")[:3]))
