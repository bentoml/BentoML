from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType
from typing import cast

from packaging.version import Version

__all__ = [
    "PackageNotFoundError",
    "pkg_version_info",
    "get_pkg_version",
    "source_locations",
    "find_spec",
]

get_pkg_version = importlib.metadata.version
find_spec = importlib.util.find_spec


@lru_cache(maxsize=None)
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


DEFAULT_BENTOML_GIT_URL = "https://github.com/bentoml/bentoml.git"


@lru_cache(maxsize=1)
def get_local_bentoml_dependency() -> str:
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "Getting local bentoml dependency, please make sure all local changes are pushed"
    )
    src_dir = Path(__file__).parent.parent.parent.parent.parent
    try:
        branch_name = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=src_dir,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        branch_name = "main"
    # Precise checkout only returns "HEAD" as branch name
    if branch_name == "HEAD":
        ref = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=src_dir,
            text=True,
        ).strip()
        remote = "origin"
    else:
        ref = branch_name
        try:
            remote = subprocess.check_output(
                ["git", "config", "--get", f"branch.{branch_name}.remote"],
                cwd=src_dir,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            remote = "origin"

    try:
        remote_url = subprocess.check_output(
            ["git", "config", "--get", f"remote.{remote}.url"],
            cwd=src_dir,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        remote_url = DEFAULT_BENTOML_GIT_URL

    return f"git+{remote_url}@{ref}"
