from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

LATEST_PROTOCOL_VERSION = "v1"


def import_generated_stubs(
    version: str = LATEST_PROTOCOL_VERSION,
    file: str = "service.proto",
) -> tuple[types.ModuleType, types.ModuleType]:
    """
    Import generated stubs.

    Args:
        version: The version of the proto file to import.
        file: The name of the proto file to import.

    Returns:
        A tuple of the generated stubs for the proto file.

    Examples:

        .. code-block:: python

           from bentoml.grpc.utils import import_generated_stubs

           # given proto file bentoml/grpc/v1/service.proto exists
           pb, services = import_generated_stubs(version="v1", file="service.proto")
    """
    # generate git root from this file's path
    from bentoml._internal.utils.lazy_loader import LazyLoader

    exception_message = f"Generated stubs for '{version}/{file}' are missing (broken installation). Please reinstall bentoml: 'pip install bentoml[grpc].'"
    file = file.split(".")[0]

    service_pb2 = LazyLoader(
        f"{file}_pb2",
        globals(),
        f"bentoml.grpc.{version}.{file}_pb2",
        exc_msg=exception_message,
    )
    service_pb2_grpc = LazyLoader(
        f"{file}_pb2_grpc",
        globals(),
        f"bentoml.grpc.{version}.{file}_pb2_grpc",
        exc_msg=exception_message,
    )
    return service_pb2, service_pb2_grpc


def import_grpc() -> tuple[types.ModuleType, types.ModuleType]:
    from bentoml._internal.utils.lazy_loader import LazyLoader

    exception_message = "'grpcio' is required for gRPC support. Install with 'pip install bentoml[grpc]'."
    grpc = LazyLoader("grpc", globals(), "grpc", exc_msg=exception_message)
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exception_message)
    return grpc, aio
