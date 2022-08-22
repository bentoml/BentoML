from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import types


def import_generated_stubs(
    version: str = "v1",
) -> tuple[types.ModuleType, types.ModuleType]:
    """
    Import generated stubs.
    """
    # generate git root from this file's path
    from bentoml._internal.utils.lazy_loader import LazyLoader

    GIT_ROOT = Path(__file__).parent.parent.parent.parent

    exception_message = f"Generated stubs are missing. To generate stubs, run '.{GIT_ROOT}/scripts/generate_grpc_stubs.sh'"

    service_pb2 = LazyLoader(
        "service_pb2",
        globals(),
        f"bentoml.grpc.{version}.service_pb2",
        exc_msg=exception_message,
    )
    service_pb2_grpc = LazyLoader(
        "service_pb2_grpc",
        globals(),
        f"bentoml.grpc.{version}.service_pb2_grpc",
        exc_msg=exception_message,
    )
    return service_pb2, service_pb2_grpc
