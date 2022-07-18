from __future__ import annotations

import os
import ast
import sys
import subprocess
from pathlib import Path

import pkg_resources
from setuptools import setup

if sys.version_info[:2] < (3, 9):
    from astunparse import unparse  # type: ignore (unfinished type)
else:
    from ast import unparse

GIT_ROOT = Path(os.path.abspath(__file__)).parent

IMPORT_PATH = "bentoml.protos"

proto_path = GIT_ROOT.joinpath("protos")

gen_stub_path: str = os.path.join(GIT_ROOT, "bentoml", "protos")

proto_include = pkg_resources.resource_filename("grpc_tools", "_proto")


# loosely defined imports fix
# we assume that the generated stubs are as follow: import stmt_pb2 as stmt__pb2
def fix_import_name(fix_name: str, node: ast.AST) -> ast.AST:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == fix_name:
                alias.name = f"{IMPORT_PATH}.{alias.name}"
    return node


def fix_imports(file: str, fix_name: str) -> None:
    with open(os.path.join(gen_stub_path, file), "r") as f:
        content = ast.parse(f.read())
    fixed_ast: list[ast.AST] = [
        fix_import_name(fix_name, node) for node in content.body
    ]
    # monkey patch the parsed content
    object.__setattr__(content, "body", fixed_ast)

    with open(os.path.join(gen_stub_path, file), "w") as f:
        f.write(unparse(content))


def gen_args(file: str, *, grpc_out: bool = False) -> list[str]:
    args = [
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
    subprocess.check_call(
        [sys.executable, "-m", "grpc_tools.protoc", *gen_args("payload")]
    )
    subprocess.check_call(
        [sys.executable, "-m", "grpc_tools.protoc", *gen_args("service", grpc_out=True)]
    )

    # fix service_pb2 imports
    fix_imports("service_pb2.py", "payload_pb2")
    fix_imports("service_pb2_grpc.py", "service_pb2")

    # run setuptools.setup()
    setup()
