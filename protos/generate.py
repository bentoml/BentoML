from __future__ import annotations

import os
import sys
import typing as t
import logging
import subprocess
from pathlib import Path

from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml._internal.utils.pkg import pkg_version_info
from bentoml._internal.utils.pkg import PackageNotFoundError

GIT_ROOT = Path(__file__).parent.parent

gen_stub_path: str = os.path.join(GIT_ROOT, "bentoml", "protos")

logger = logging.getLogger("bentoml")


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
        raise BentoMLException(f"Path {file_path.__fspath__()} does not exist.")

    cmd.extend(gen_args(proto_path, filename, **kwargs))
    logger.debug("Running: %s", cmd)

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise BentoMLException(e.stderr.decode("utf-8")) from e


def main() -> int:
    try:
        _ = pkg_version_info("grpcio-tools")
    except PackageNotFoundError:
        raise MissingDependencyException(
            "grpcio_tools is required to generate proto stubs. Install with `pip install grpcio-tools`."
        )

    try:
        # generate io_descriptors_pb2.py
        run_protoc("payload")

        # generate service_pb2.py and service_grpc_pb2.py
        run_protoc("service", grpc_out=True)
        logger.info(f"Successfully generated gRPC stubs to {gen_stub_path}.")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
