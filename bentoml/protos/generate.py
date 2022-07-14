from __future__ import annotations

import os
import sys
import typing as t
import logging
import argparse
import subprocess
from pathlib import Path

from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml._internal.utils.pkg import pkg_version_info
from bentoml._internal.utils.pkg import PackageNotFoundError

DEFAULT_VERSION = "v1alpha1"

parent_dir = Path(__file__).parent

logger = logging.getLogger("bentoml")


def gen_args(
    proto_path: str | Path,
    protofile: str,
    out_path: str | Path,
    *,
    grpc_out: bool = False,
) -> list[str]:
    args: list[str] = [f"-I{str(proto_path)}", f"--python_out={(out_path)}"]
    if grpc_out:
        args.append(f"--grpc_python_out={(out_path)}")
    args.append(os.path.join(proto_path, protofile))
    return args


def run_protoc(
    package: str, filename: str, args: argparse.Namespace, **kwargs: t.Any
) -> None:
    """Run protoc with given package and args."""
    logger.info("Generating protobuf stub for %s", package)
    cmd = [sys.executable, "-m", "grpc_tools.protoc"]

    proto_path = parent_dir.joinpath(package, args.version)
    if not filename.endswith(".proto"):
        filename = f"{filename}.proto"

    if not proto_path.exists():
        raise BentoMLException(f"Path {proto_path.__fspath__()} does not exist")

    cmd.extend(gen_args(proto_path, filename, parent_dir, **kwargs))
    logger.debug("Running: %s", cmd)

    try:
        proc = subprocess.run(cmd, capture_output=True)
        proc.check_returncode()
    except subprocess.CalledProcessError as e:
        raise BentoMLException(e.stderr.decode("utf-8")) from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate gRPC stubs.")
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        metavar="version",
        help=f"version (default: {DEFAULT_VERSION})",
    )
    args = parser.parse_args()

    try:
        _ = pkg_version_info("grpcio-tools")
    except PackageNotFoundError:
        raise MissingDependencyException(
            "grpcio_tools is required to generate proto stubs. Install with `pip install grpcio-tools`."
        )

    try:
        # generate io_descriptors_pb2.py
        run_protoc("service", "io_descriptors", args)

        # generate service_pb2.py and service_grpc_pb2.py
        run_protoc("service", "service", args, grpc_out=True)
        logger.info(f"Successfully generated gRPC stubs to {parent_dir}.")
        return 0
    except Exception as e:
        logger.error(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
