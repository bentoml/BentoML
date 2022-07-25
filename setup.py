from __future__ import annotations

import os
import sys
import typing as t
import subprocess
import importlib.util
from pathlib import Path

from setuptools import setup

GIT_ROOT = Path(os.path.abspath(__file__)).parent

_VERSION_MAP = {
    "v1": {
        ("service.proto", "service_test.proto"): {"grpc_out": True},
        ("struct.proto",): {},
    },
}


def source_locations(pkg: str) -> str:
    spec = importlib.util.find_spec(pkg)
    if not spec:
        raise RuntimeError(
            f"{pkg} not found. Make sure to add to both pyproject.toml and setup.cfg."
        )
    (location,) = spec.submodule_search_locations  # type: ignore (unfinished type)
    return t.cast(str, location)


def get_args(parent_path: str, *paths: str, grpc_out: bool = False) -> list[str]:
    args = [
        "-I.",
        f"-I{os.path.dirname(source_locations('google'))}",  # include common googleapis stubs.
        "--python_out=.",
        "--mypy_out=.",
    ]
    if grpc_out:
        args.extend(["--grpc_python_out=.", "--mypy_grpc_out=."])
    args.extend([os.path.join(parent_path, path) for path in paths])
    return args


if __name__ == "__main__":

    # run setuptools.setup()
    setup()

    binary = [sys.executable, "-m", "grpc_tools.protoc"]

    for version, file_map in _VERSION_MAP.items():
        version_path = os.path.join("bentoml", "grpc", version)

        # Generate bentoml stubs
        for paths, options in file_map.items():
            subprocess.check_call([*binary, *get_args(version_path, *paths, **options)])
