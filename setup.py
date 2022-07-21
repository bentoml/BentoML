from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup

GIT_ROOT = Path(os.path.abspath(__file__)).parent

VERSION = "v1"


if __name__ == "__main__":

    # run setuptools.setup()
    setup()

    binary = [sys.executable, "-m", "grpc_tools.protoc"]

    # Generate bentoml stubs
    for path in ["service.proto", "service_test.proto"]:
        subprocess.check_call(
            [
                *binary,
                "-I.",
                "--python_out=.",
                "--mypy_out=.",
                "--grpc_python_out=.",
                "--mypy_grpc_out=.",
                os.path.join("bentoml", "grpc", VERSION, path),
            ]
        )
