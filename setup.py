from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pkg_resources
from setuptools import setup

GIT_ROOT = Path(os.path.abspath(__file__)).parent

proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")


def gen_args(
    file: str,
    *,
    directory: list[str] | str = "grpc",
    grpc_out: bool = False,
) -> list[str]:
    args = [
        "-I.",
        # add verbatim sources
        f"-I{os.path.join('.','bentoml', 'grpc', 'verbatim')}",
        "--python_out=.",
        "--mypy_out=.",
    ]
    if grpc_out:
        args.extend(["--grpc_python_out=.", "--mypy_grpc_out=."])

    if not file.endswith(".proto"):
        file += ".proto"
    if not isinstance(directory, list):
        directory = [directory]
    file_path = os.path.join("bentoml", *directory, file)

    args.append(file_path)

    return args


if __name__ == "__main__":

    # run setuptools.setup()
    setup()

    binary = [sys.executable, "-m", "grpc_tools.protoc"]

    # Generate bentoml stubs
    subprocess.check_call([*binary, *gen_args("payload")])
    subprocess.check_call([*binary, *gen_args("service", grpc_out=True)])

    # Generate verbatim stubs
    subprocess.check_call(
        [*binary, *gen_args("status", directory=["grpc", "verbatim"])]
    )
    subprocess.check_call([*binary, *gen_args("code", directory=["grpc", "verbatim"])])
