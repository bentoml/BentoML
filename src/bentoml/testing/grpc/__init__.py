from __future__ import annotations

import typing as t
import importlib
import traceback
from typing import TYPE_CHECKING
from contextlib import ExitStack
from contextlib import asynccontextmanager

from ...exceptions import BentoMLException
from ...grpc.utils import import_grpc
from ...grpc.utils import import_generated_stubs
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..._internal.utils import LazyLoader
from ..._internal.utils import reserve_free_port
from ..._internal.utils import cached_contextmanager
from ..._internal.utils import add_experimental_docstring

if TYPE_CHECKING:
    import grpc
    import numpy as np
    from grpc import aio
    from numpy.typing import NDArray
    from grpc.aio._channel import Channel
    from google.protobuf.message import Message

    from ...grpc.v1 import service_pb2 as pb
    from ..._internal.service import Service
else:
    grpc, aio = import_grpc()  # pylint: disable=E1111
    np = LazyLoader("np", globals(), "numpy")

__all__ = [
    "async_client_call",
    "randomize_pb_ndarray",
    "make_pb_ndarray",
    "create_channel",
    "make_standalone_server",
    "create_test_bento_servicer",
]


def create_test_bento_servicer(
    service: Service,
    protocol_version: str = LATEST_PROTOCOL_VERSION,
) -> t.Callable[[Service], t.Any]:
    try:
        module = importlib.import_module(
            f".{protocol_version}", package="bentoml._internal.server.grpc.servicer"
        )
        return getattr(module, "create_bento_servicer")(service)
    except (ImportError, ModuleNotFoundError):
        raise BentoMLException(
            f"Failed to load servicer implementation for version {protocol_version}"
        ) from None


def randomize_pb_ndarray(
    shape: tuple[int, ...], protocol_version: str = LATEST_PROTOCOL_VERSION
) -> pb.NDArray:
    pb, _ = import_generated_stubs(protocol_version)
    arr: NDArray[np.float32] = t.cast("NDArray[np.float32]", np.random.rand(*shape))
    return pb.NDArray(
        shape=list(shape), dtype=pb.NDArray.DTYPE_FLOAT, float_values=arr.ravel()
    )


def make_pb_ndarray(
    arr: NDArray[t.Any], protocol_version: str = LATEST_PROTOCOL_VERSION
) -> pb.NDArray:
    from bentoml._internal.io_descriptors.numpy import npdtype_to_dtypepb_map
    from bentoml._internal.io_descriptors.numpy import npdtype_to_fieldpb_map

    pb, _ = import_generated_stubs(protocol_version)

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
    sanity: bool = True,
    timeout: int | None = 90,
    assert_data: pb.Response | t.Callable[[pb.Response], bool] | None = None,
    assert_code: grpc.StatusCode | None = None,
    assert_details: str | None = None,
    assert_trailing_metadata: aio.Metadata | None = None,
    protocol_version: str = LATEST_PROTOCOL_VERSION,
) -> pb.Response | None:
    """
    Invoke a given API method via a client.

    Args:
        method: The method name to call.
        channel: The given aio.Channel to use for invoking the RPC. Channels shouldn't include
                 any client interceptors. as we will handle interceptors' lifecycle separately.
        data: The data to send to the server.
        assert_data: The data to assert against the response.
        timeout: The timeout for the RPC.
        sanity: Whether to perform sanity check on the response.
        assert_code: The code to assert against the response.
        assert_details: The details to assert against the response.

    Returns:
        The response from the server.
    """
    pb, _ = import_generated_stubs(protocol_version)

    res: pb.Response | None = None
    try:
        Call = channel.unary_unary(
            f"/bentoml.grpc.{protocol_version}.BentoService/Call",
            request_serializer=pb.Request.SerializeToString,
            response_deserializer=pb.Response.FromString,
        )
        output: aio.UnaryUnaryCall[pb.Request, pb.Response] = Call(
            pb.Request(api_name=method, **data), timeout=timeout
        )
        res = await t.cast(t.Awaitable[pb.Response], output)
        return_code = await output.code()
        details = await output.details()
        trailing_metadata = await output.trailing_metadata()
        if sanity:
            assert isinstance(res, pb.Response)
        if assert_data:
            if callable(assert_data):
                assert assert_data(res), f"Failed while checking data: {output}"
            else:
                assert res == assert_data, f"Failed while checking data: {output}"
    except aio.AioRpcError as call:
        return_code = call.code()
        details = call.details()
        trailing_metadata = call.trailing_metadata()
    if assert_code is not None:
        assert (
            return_code == assert_code
        ), f"Method '{method}' returns {return_code} while expecting {assert_code}."
    if assert_details is not None:
        assert (
            assert_details == details
        ), f"Details '{assert_details}' is not in '{details}'."
    if assert_trailing_metadata is not None:
        assert (
            trailing_metadata == assert_trailing_metadata
        ), f"Trailing metadata '{trailing_metadata}' while expecting '{assert_trailing_metadata}'."
    return res


@asynccontextmanager
@add_experimental_docstring
async def create_channel(
    host_url: str,
    interceptors: t.Sequence[aio.ClientInterceptor] | None = None,
) -> t.AsyncGenerator[Channel, None]:
    """
    Create an async channel with given host_url and client interceptors.

    Args:
        host_url: The host url to connect to.
        interceptors: The client interceptors to use. This is optional, by default set to None.

    Returns:
        A insecure channel.
    """
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
    server.add_insecure_port(f"{host}:{port}")
    print("Using port %d..." % port)
    try:
        yield server, "localhost:%d" % port
    finally:
        stack.close()
