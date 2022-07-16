from __future__ import annotations

import os
import sys
import typing as t
import logging
import traceback
import subprocess
from pathlib import Path

import setuptools

GIT_ROOT = Path(os.path.abspath(__file__)).parent

gen_stub_path: str = os.path.join(GIT_ROOT, "bentoml", "protos")

logger = logging.getLogger(__name__)


def gen_args(
    proto_path: str | Path,
    protofile: str,
    *,
    grpc_out: bool = False,
) -> list[str]:
    args: list[str] = [f"-I{str(proto_path)}", f"--python_out={gen_stub_path}"]
    if grpc_out:
        args.append(f"--grpc_python_out={gen_stub_path}")
    args.append(os.path.join(proto_path, protofile))
    return args


def run_protoc(filename: str, **kwargs: t.Any) -> None:
    # Run protoc for the given filename.
    logger.info("Generating protobuf stub for %s", filename)
    cmd = [sys.executable, "-m", "grpc_tools.protoc"]

    proto_path = GIT_ROOT.joinpath("protos")
    if not filename.endswith(".proto"):
        filename = f"{filename}.proto"
    file_path = proto_path.joinpath(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Path {file_path.__fspath__()} does not exist.")

    cmd.extend(gen_args(proto_path, filename, **kwargs))
    logger.debug("Running: %s", cmd)

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        # generate io_descriptors_pb2.py
        run_protoc("payload")

        # generate service_pb2.py and service_grpc_pb2.py
        run_protoc("service", grpc_out=True)
        logger.info(f"Successfully generated gRPC stubs to {gen_stub_path}.")

        # run setuptools.build_meta
        sys.exit(setuptools.setup())
    except subprocess.CalledProcessError as e:
        logger.error(e)
        sys.exit(1)
