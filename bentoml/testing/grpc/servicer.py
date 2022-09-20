# pylint: disable=unused-argument
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grpc import aio

    from bentoml.grpc.v1alpha1 import service_test_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_test_pb2_grpc as services
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, services = import_generated_stubs(file="service_test.proto")


class TestServiceServicer(services.TestServiceServicer):
    async def Execute(  # type: ignore (no async types) # pylint: disable=invalid-overridden-method
        self,
        request: pb.ExecuteRequest,
        context: aio.ServicerContext[pb.ExecuteRequest, pb.ExecuteResponse],
    ) -> pb.ExecuteResponse:
        return pb.ExecuteResponse(output="Hello, {}!".format(request.input))
