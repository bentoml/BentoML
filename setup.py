from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pkg_resources
from setuptools import setup

GIT_ROOT = Path(os.path.abspath(__file__)).parent

proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")


def gen_args(file: str, *, grpc_out: bool = False) -> list[str]:
    args = ["-I.", "--python_out=.", "--mypy_out=."]
    if grpc_out:
        args.extend(["--grpc_python_out=.", "--mypy_grpc_out=."])

    if not file.endswith(".proto"):
        file += ".proto"
    file_path = os.path.join("bentoml", "protos", file)

    args.append(file_path)

    return args


if __name__ == "__main__":

    # run setuptools.setup()
    setup()

    # Run before setup
    subprocess.check_call(
        [sys.executable, "-m", "grpc_tools.protoc", *gen_args("payload")]
    )
    subprocess.check_call(
        [sys.executable, "-m", "grpc_tools.protoc", *gen_args("service", grpc_out=True)]
    )
