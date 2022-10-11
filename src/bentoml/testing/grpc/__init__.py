from __future__ import annotations

import typing as t
import traceback
from typing import TYPE_CHECKING
from contextlib import ExitStack
from contextlib import asynccontextmanager

from bentoml.exceptions import BentoMLException
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils import cached_contextmanager
from bentoml._internal.utils import add_experimental_docstring
from bentoml._internal.server.grpc.servicer import create_bento_servicer

from .servicer import TestServiceServicer

if TYPE_CHECKING:
    import grpc
    import numpy as np
    from grpc import aio
    from numpy.typing import NDArray
    from grpc.aio._channel import Channel
    from google.protobuf.message import Message

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
    from bentoml.grpc.v1alpha1 import service_test_pb2_grpc as services_test
else:
    from bentoml.grpc.utils import import_grpc
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()
    _, services_test = import_generated_stubs(file="service_test.proto")
    grpc, aio = import_grpc()  # pylint: disable=E1111
    np = LazyLoader("np", globals(), "numpy")

__all__ = [
    "async_client_call",
    "randomize_pb_ndarray",
    "make_pb_ndarray",
    "create_channel",
    "make_standalone_server",
    "TestServiceServicer",
    "create_bento_servicer",
]


def randomize_pb_ndarray(shape: tuple[int, ...]) -> pb.NDArray:
    arr: NDArray[np.float32] = t.cast("NDArray[np.float32]", np.random.rand(*shape))
    return pb.NDArray(
        shape=list(shape), dtype=pb.NDArray.DTYPE_FLOAT, float_values=arr.ravel()
    )


def make_pb_ndarray(arr: NDArray[t.Any]) -> pb.NDArray:
    from bentoml._internal.io_descriptors.numpy import npdtype_to_dtypepb_map
    from bentoml._internal.io_descriptors.numpy import npdtype_to_fieldpb_map

    try:
        fieldpb = npdtype_to_fieldpb_map()[arr.dtype]
        dtypepb = npdtype_to_dtypepb_map()[arr.dtype]
        return pb.NDArray(
            **{
                fieldpb: arr.ravel().tolist(),
                "dtype": dtypepb,
                "shape": tuple(arr.shape),
            },
        )
    except KeyError:
        raise BentoMLException(
            f"Unsupported dtype '{arr.dtype}' for response message.",
        ) from None


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
        channel: The given aio.Channel to use for invoking the RPC. Channels shouldn't include
                 any client interceptors. as we will handle interceptors' lifecycle separately.
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
    # We will add our own interceptors to the channel, which means
    # We will have to check whether channel already has interceptors.
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
@add_experimental_docstring
async def create_channel(
    host_url: str, interceptors: t.Sequence[aio.ClientInterceptor] | None = None
) -> t.AsyncGenerator[Channel, None]:
    """Create an async channel with given host_url and client interceptors."""
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


@add_experimental_docstring
@cached_contextmanager("{interceptors}")
def make_standalone_server(
    interceptors: t.Sequence[aio.ServerInterceptor] | None = None,
    host: str = "0.0.0.0",
) -> t.Generator[tuple[aio.Server, str], None, None]:
    """
    Create a standalone aio.Server for testing.

    Args:
        interceptors: The interceptors to use for the server, default to None.

    Returns:
        The server and the host_url.

    Example for async test cases:

    .. code-block:: python

        async def test_some_async():
            with make_standalone_server() as (server, host_url):
                try:
                    await server.start()
                    channel = grpc.aio.insecure_channel(host_url)
                    ...  # test code here
                finally:
                    await server.stop(None)

    Example for sync test cases:

    .. code-block:: python

        def test_cases():
            import asyncio

            loop = asyncio.new_event_loop()
            with make_standalone_server() as (server, host_url):
                try:
                    loop.run_until_complete(server.start())
                    channel = grpc.insecure_channel(host_url)
                    ...  # test code here
                finally:
                    loop.call_soon_threadsafe(lambda: asyncio.ensure_future(server.stop(None)))
                    loop.close()
                assert loop.is_closed()
    """
    stack = ExitStack()
    port = stack.enter_context(reserve_free_port(enable_so_reuseport=True))
    server = aio.server(
        interceptors=interceptors,
        options=(("grpc.so_reuseport", 1),),
    )
    services_test.add_TestServiceServicer_to_server(TestServiceServicer(), server)  # type: ignore (no async types) # pylint: disable=E0601
    server.add_insecure_port(f"{host}:{port}")
    print("Using port %d..." % port)
    try:
        yield server, "localhost:%d" % port
    finally:
        stack.close()
