from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import grpc
    from grpc import aio

    from bentoml.grpc.types import Request
    from bentoml.grpc.types import BentoUnaryUnaryCall
else:
    aio = LazyLoader("aio", globals(), "grpc.aio")


class AssertClientInterceptor(aio.UnaryUnaryClientInterceptor):
    def __init__(
        self,
        assert_code: grpc.StatusCode | None = None,
        assert_details: str | None = None,
        assert_trailing_metadata: aio.Metadata | None = None,
    ):
        self._assert_code = assert_code
        self._assert_details = assert_details
        self._assert_trailing_metadata = assert_trailing_metadata

    async def intercept_unary_unary(  # type: ignore (unable to infer types from parameters)
        self,
        continuation: t.Callable[[aio.ClientCallDetails, Request], BentoUnaryUnaryCall],
        client_call_details: aio.ClientCallDetails,
        request: Request,
    ) -> BentoUnaryUnaryCall:
        # Note that we cast twice here since grpc.aio._call.UnaryUnaryCall
        # implements __await__, which returns ResponseType. However, pyright
        # are unable to determine types from given mixin.
        #
        # continuation(client_call_details, request) -> call: UnaryUnaryCall
        # await call -> ResponseType
        call = await t.cast(
            "t.Awaitable[BentoUnaryUnaryCall]",
            continuation(client_call_details, request),
        )
        try:
            code = await call.code()
            details = await call.details()
            trailing_metadata = await call.trailing_metadata()
            if self._assert_code:
                assert (
                    code == self._assert_code
                ), f"{call!r} returns {await call.code()} while expecting {self._assert_code}."
            if self._assert_details:
                assert (
                    self._assert_details in details
                ), f"'{self._assert_details}' is not in {await call.details()}."
            if self._assert_trailing_metadata:
                assert (
                    self._assert_trailing_metadata == trailing_metadata
                ), f"Trailing metadata '{trailing_metadata}' while expecting '{self._assert_trailing_metadata}'."
            return call
        except AssertionError as e:
            raise e from None
