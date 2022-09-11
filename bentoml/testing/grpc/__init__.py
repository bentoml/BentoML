from __future__ import annotations

import typing as t
import traceback
from typing import TYPE_CHECKING
from contextlib import contextmanager
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from bentoml._internal.utils import LazyLoader

from ._io import make_pb_ndarray
from ._io import randomize_pb_ndarray

if TYPE_CHECKING:
    import grpc
    from grpc import aio
    from grpc.aio._channel import Channel
    from google.protobuf.message import Message

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()
    exception_msg = (
        "'grpcio' is not installed. Please install it with 'pip install -U grpcio'"
    )
    grpc = LazyLoader("grpc", globals(), "grpc", exc_msg=exception_msg)
    aio = LazyLoader("aio", globals(), "grpc.aio", exc_msg=exception_msg)

__all__ = [
    "async_client_call",
    "randomize_pb_ndarray",
    "make_pb_ndarray",
    "create_channel",
    "make_standalone_server",
]


async def async_client_call(
    method: str,
    channel: Channel,
    data: dict[str, Message | pb.Part | bytes | str | dict[str, t.Any]],
    assert_data: pb.Response | t.Callable[[pb.Response], bool] | None = None,
    assert_code: grpc.StatusCode | None = None,
    assert_details: str | None = None,
    timeout: int | None = None,
    sanity: bool = True,
) -> pb.Response:
    """
    Note that to use this function, 'channel' should not be created with any client interceptors,
    since we will handle interceptors' lifecycle separately.

    This function will also mimic the generated stubs function 'Call' from given 'channel'.

    Args:
        method: The method name to call.
        channel: The given aio.Channel to use for invoking the RPC.
        data: The data to send to the server.
        assert_data: The data to assert against the response.
        assert_code: The code to assert against the response.
        assert_details: The details to assert against the response.
        timeout: The timeout for the RPC.
        sanity: Whether to perform sanity check on the response.

    Returns:
        The response from the server.
    """
    from bentoml.testing.grpc.interceptors import AssertClientInterceptor

    if assert_code is None:
        # by default, we want to check if the request is healthy
        assert_code = grpc.StatusCode.OK
    assert (
        len(
            list(
                filter(
                    lambda x: len(x) != 0,
                    map(
                        lambda stack: getattr(channel, stack),
                        [
                            "_unary_unary_interceptors",
                            "_unary_stream_interceptors",
                            "_stream_unary_interceptors",
                            "_stream_stream_interceptors",
                        ],
                    ),
                )
            )
        )
        == 0
    ), "'channel' shouldn't have any interceptors."
    try:
        # we will handle adding our testing interceptors here.
        # prefer not to use private attributes, but this will do
        channel._unary_unary_interceptors.append(  # type: ignore (private warning)
            AssertClientInterceptor(
                assert_code=assert_code, assert_details=assert_details
            )
        )
        Call = channel.unary_unary(
            "/bentoml.grpc.v1alpha1.BentoService/Call",
            request_serializer=pb.Request.SerializeToString,
            response_deserializer=pb.Response.FromString,
        )
        output = await t.cast(
            t.Awaitable[pb.Response],
            Call(pb.Request(api_name=method, **data), timeout=timeout),
        )
        if sanity:
            assert output
        if assert_data:
            try:
                if callable(assert_data):
                    assert assert_data(output)
                else:
                    assert output == assert_data
            except AssertionError:
                raise AssertionError(f"Failed while checking data: {output}") from None
        return output
    finally:
        # we will reset interceptors per call
        channel._unary_unary_interceptors = []  # type: ignore (private warning)


@asynccontextmanager
async def create_channel(
    host_url: str, interceptors: t.Sequence[aio.ClientInterceptor] | None = None
) -> t.AsyncGenerator[Channel, None]:
    channel: Channel | None = None
    try:
        async with aio.insecure_channel(host_url, interceptors=interceptors) as channel:
            # create a blocking call to wait til channel is ready.
            await channel.channel_ready()
            yield channel
    except aio.AioRpcError as e:
        traceback.print_exc()
        raise e from None
    finally:
        if channel:
            await channel.close()


@contextmanager
def make_standalone_server(
    bind_address: str, interceptors: t.Sequence[aio.ServerInterceptor] | None = None
) -> t.Generator[aio.Server, None, None]:
    server = aio.server(
        interceptors=interceptors,
        migration_thread_pool=ThreadPoolExecutor(max_workers=1),
        options=(("grpc.so_reuseport", 0),),
    )
    server.add_insecure_port(bind_address)
    yield server
