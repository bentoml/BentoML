from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup

GIT_ROOT = Path(os.path.abspath(__file__)).parent

_VERSION_MAP = {
    "v1": {
        ("service.proto", "service_test.proto"): {"grpc_out": True},
    },
}


def get_args(parent_path: str, *paths: str, grpc_out: bool = False) -> list[str]:
    args = [
        "-I.",
        "--python_out=.",
        "--mypy_out=.",
    ]
    if grpc_out:
        args.extend(["--grpc_python_out=.", "--mypy_grpc_out=."])
    args.extend([os.path.join(parent_path, path) for path in paths])
    return args


if __name__ == "__main__":

    binary = [sys.executable, "-m", "grpc_tools.protoc"]

    for version, file_map in _VERSION_MAP.items():
        version_path = os.path.join("bentoml", "grpc", version)

        # Generate bentoml stubs
        for paths, options in file_map.items():
            subprocess.check_call([*binary, *get_args(version_path, *paths, **options)])

    # run setuptools.setup()
    setup()
