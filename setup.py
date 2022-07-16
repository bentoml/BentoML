from __future__ import annotations

import os
from pathlib import Path

import pkg_resources
from setuptools import setup
from grpc_tools.protoc import main as run_main  # type: ignore (unfinished type definition)

GIT_ROOT = Path(os.path.abspath(__file__)).parent

proto_path = GIT_ROOT.joinpath("protos")

gen_stub_path: str = os.path.join(GIT_ROOT, "bentoml", "protos")

proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")


def gen_args(file: str, *, grpc_out: bool = False) -> list[str]:
    args = [
        f"-I{proto_include}",
        f"-I{str(proto_path)}",
        f"--python_out={gen_stub_path}",
    ]
    if grpc_out:
        args.append(f"--grpc_python_out={gen_stub_path}")

    if not file.endswith(".proto"):
        file += ".proto"
    file_path = proto_path.joinpath(file)
    if not file_path.exists():
        raise FileNotFoundError(f"{str(file_path)} not found")

    args.append(file_path.__fspath__())

    return args


if __name__ == "__main__":

    # Run before setup
    run_main(gen_args("payload"))
    run_main(gen_args("service", grpc_out=True))

    # run setuptools.setup()
    setup()
