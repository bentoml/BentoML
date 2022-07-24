from __future__ import annotations

import typing as t
import asyncio
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING
from functools import wraps

import grpc

from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grpc import aio

    from bentoml.grpc.v1.service_pb2 import CallRequest
    from bentoml.grpc.v1.service_pb2 import CallResponse
    from bentoml.grpc.v1.service_pb2 import ServerLiveRequest
    from bentoml.grpc.v1.service_pb2 import ServerLiveResponse
    from bentoml.grpc.v1.service_pb2 import ServerReadyRequest
    from bentoml.grpc.v1.service_pb2 import ServerReadyResponse
    from bentoml.grpc.v1.service_pb2_grpc import BentoServiceServicer

    ResponseType = CallResponse | ServerLiveResponse | ServerReadyResponse
    RequestType = CallRequest | ServerLiveRequest | ServerReadyRequest
    BentoServicerContext = aio.ServicerContext[ResponseType, RequestType]

_STATUS_CODE_MAPPING = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
    HTTPStatus.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    HTTPStatus.UNPROCESSABLE_ENTITY: grpc.StatusCode.FAILED_PRECONDITION,
}


def grpc_status_code(err: BentoMLException) -> grpc.StatusCode:
    """
    Convert BentoMLException.error_code to grpc.StatusCode.
    """
    return _STATUS_CODE_MAPPING.get(err.error_code, grpc.StatusCode.UNKNOWN)


def handle_grpc_error(fn: t.Callable[..., t.Any]) -> t.Any:
    """
    Decorator to handle grpc error.
    """

    @wraps(fn)
    async def wrapper(
        self: BentoServiceServicer,
        request: RequestType,
        context: BentoServicerContext,
    ) -> ResponseType | None:
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(self, request, context)
            else:
                return fn(self, request, context)
        except BentoMLException as e:
            logger.error(e)
            await context.abort(code=grpc_status_code(e), details=str(e))

    return wrapper
