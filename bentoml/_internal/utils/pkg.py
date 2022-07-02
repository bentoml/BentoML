from __future__ import annotations

from typing import cast

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

import importlib.util

get_pkg_version = importlib_metadata.version


def source_locations(pkg: str) -> str | None:
    module = importlib.util.find_spec(pkg)
    if module is None:
        return
    (module_path,) = module.submodule_search_locations  # type: ignore (unfinished typed)
    return cast(str, module_path)
