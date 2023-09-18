from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest

from bentoml._internal.utils.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import grpc
    from grpc import aio
else:
    from bentoml.grpc.utils import import_grpc

    grpc, _ = import_grpc()


@pytest.fixture(scope="module", name="mock_unary_unary_handler")
def fixture_mock_handler() -> MagicMock:
    handler = MagicMock(spec=grpc.RpcMethodHandler)
    handler.request_streaming = PropertyMock(return_value=False)
    handler.response_streaming = PropertyMock(return_value=False)
    return handler


if TYPE_CHECKING:
    from tests.proto import service_test_pb2 as pb
    from tests.proto import service_test_pb2_grpc as services
else:
    pb = LazyLoader("pb", globals(), "tests.proto.service_test_pb2")
    services = LazyLoader("services", globals(), "tests.proto.service_test_pb2_grpc")


class TestServiceServicer(services.TestServiceServicer):
    async def Execute(  # type: ignore (no async types) # pylint: disable=invalid-overridden-method
        self,
        request: pb.ExecuteRequest,
        context: aio.ServicerContext[pb.ExecuteRequest, pb.ExecuteResponse],
    ) -> pb.ExecuteResponse:
        return pb.ExecuteResponse(output="Hello, {}!".format(request.input))
