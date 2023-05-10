from __future__ import annotations

import enum
import typing as t
import asyncio
import logging
import functools
from abc import ABC
from abc import abstractmethod

from ...exceptions import BentoMLException
from ..service.inference_api import InferenceAPI
from ..io_descriptors.multipart import Multipart

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from types import TracebackType

    from .grpc import GrpcClient
    from .grpc import AsyncGrpcClient
    from .http import HTTPClient
    from .http import AsyncHTTPClient
    from ..service import Service

    P = t.ParamSpec("P")
    F = t.Callable[P, t.Any]

    ClientKind = t.Literal[
        "auto", "http", "grpc", "async_http", "async_grpc", "async-grpc", "async-http"
    ]

    class ConnectionProtocol(t.Protocol):
        def close(self) -> t.Any:
            ...

    class ClientProtocol(t.Protocol):
        server_url: str
        endpoints: list[str]
        supports_kwds_assignment: bool

        _svc: Service
        _conn_type: t.Any
        _endpoint_kwds_map: dict[str, list[str]] | None

        def __init__(self, svc: Service, server_url: str):
            ...

        def _getitem_from_kwds_map(self, api: InferenceAPI) -> list[str]:
            ...


def ensure_exec_coro(coro: t.Coroutine[t.Any, t.Any, t.Any]) -> t.Any:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    else:
        return loop.run_until_complete(coro)


# NOTE: The purpose of this function is to wraps auto generated client function
# with nice docstring and better naming.
def wraps_call_attributes(
    f: t.Callable[P, t.Any],
    api: InferenceAPI,
    svc: Service,
    *,
    _func_name: str | None = None,
    _async_doc: bool = False,
) -> t.Any:
    if _func_name is None:
        _func_name = f.func.__name__ if isinstance(f, functools.partial) else f.__name__

    def _invoke(_: t.Any, *args: P.args, **kwargs: P.kwargs) -> t.Any:
        return f(*args, **kwargs)

    def _repr(self: t.Any) -> str:
        return f"<generated client function '{_func_name}' of {svc.name}>"

    input_type = api.input.input_type()
    if api.multi_input:
        # NOTE: Right now, only Multipart.input_type returns a dict
        assert isinstance(input_type, dict) and isinstance(api.input, Multipart)
        inputs = api.input._inputs
        args = "\n".join(
            [f"{key}: data with type {value}." for key, value in input_type.items()]
        )
        examples = (
            f"result ={' await' if _async_doc else ''} client.{_func_name}("
            + ", ".join(
                [
                    f"{key}={value.sample if value.sample else '<input>'}"
                    for key, value in zip(input_type, inputs.values())
                ]
            )
            + ")"
        )
    else:
        args = f"inp: data with type {input_type}. This can be used either as keyword arguments or the first positional argument to {_func_name}"
        examples = f"""result = {'await' if _async_doc else ''} client.{_func_name}({api.input.sample if api.input.sample else '<input>'})"""

    f_docs = f"""\
            Sending request to '{_func_name}' from service '{svc.name}'.

            Usage for using '{_func_name}' endpoint with ``bentoml.client.Client``:

            .. code-block:: python

                {examples}

            .. note::

                See https://docs.bentoml.org/en/latest/reference/api_io_descriptors.html for more details on specific inputs/outputs types.

            Args:
                {args}

            Returns:
                data with type {api.output.input_type()}.
    """

    return type(
        "_generated_" + _func_name,
        (),
        {
            "__name__": _func_name,
            "__call__": _invoke,
            "__doc__": f_docs,
            "__repr__": _repr,
        },
    )()


class ClientType(enum.Enum):
    ASYNC = "async"
    SYNC = "sync"

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, ClientType):
            return self.value == other.value
        return False


class Client(ABC):
    server_url: str
    endpoints: list[str]

    _svc: Service

    # _conn_type represents a Connection protocol that can be used
    # to connect to the server. On gRPC, it is grpc.Channel, on HTTP, it is
    # either requests.Session or aiohttp.ClientSession
    _conn_type: ConnectionProtocol | None = None

    _CLIENT_TYPE: ClientType

    def __init_subclass__(
        cls, *, client_type: t.Literal["sync", "async"] = "sync"
    ) -> None:
        try:
            cls._CLIENT_TYPE = ClientType[client_type.upper()]
        except KeyError:
            raise BentoMLException(
                f"client_type should be either 'sync' or 'async', got {client_type}"
            )

    def __init__(self, svc: Service, server_url: str):
        self._svc = svc
        self.server_url = server_url

        if svc is not None and len(svc.apis) == 0:
            raise BentoMLException("No APIs were found when constructing client.")

        self.endpoints = []
        for name in self._svc.apis:
            self.endpoints.append(name)

    @staticmethod
    def wait_until_server_ready(
        host: str, port: int, timeout: float = 30, **kwargs: t.Any
    ) -> None:
        try:
            from .http import HTTPClientMixin

            HTTPClientMixin.wait_until_server_ready(host, port, timeout, **kwargs)
        except Exception:
            try:
                # when address is a RPC
                from .grpc import GrpcClientMixin

                GrpcClientMixin.wait_until_server_ready(host, port, timeout, **kwargs)
            except Exception as err:
                # caught all other exceptions
                logger.error("Failed to connect to server %s:%s", host, port)
                logger.error(err)
                raise

    @t.overload
    @staticmethod
    def from_url(
        server_url: str, *, kind: None | t.Literal["auto"] = ...
    ) -> GrpcClient | HTTPClient | AsyncGrpcClient | AsyncHTTPClient:
        ...

    @t.overload
    @staticmethod
    def from_url(server_url: str, *, kind: t.Literal["http"] = ...) -> HTTPClient:
        ...

    @t.overload
    @staticmethod
    def from_url(
        server_url: str, *, kind: t.Literal["grpc"] = ..., **kwargs: t.Any
    ) -> GrpcClient:
        ...

    @t.overload
    @staticmethod
    def from_url(
        server_url: str, *, kind: t.Literal["async_http", "async-http"] = ...
    ) -> AsyncHTTPClient:
        ...

    @t.overload
    @staticmethod
    def from_url(
        server_url: str,
        *,
        kind: t.Literal["async_grpc", "async-grpc"] = ...,
        **kwargs: t.Any,
    ) -> AsyncGrpcClient:
        ...

    @staticmethod
    def from_url(
        server_url: str,
        *,
        kind: ClientKind | None = None,
        **kwargs: t.Any,
    ) -> Client:
        """Construct a BentoML Client from a server URL.

        Users can specify the client type by passing in the `kind` parameter.
        If `kind` is either not specified or set to 'auto', BentoML will try to construct
        a client.

        .. note:: ``'kind=auto'`` behaviour

           The client will implement a greedy check between HTTP and gRPC client. It will try to create
           a HTTP client. If the server is not a HTTP server, it will try to create a gRPC client.

           At the end of the check, if the client is not created, it will raise an exception.
           Note that it will create an async client variant if the method is constructed within a running event loop.

           :bdg-info:`Remarks:` We recommend you to explicitly specify the client type via the ``kind`` parameter to avoid breaking change.

        Args:
            server_url: The URL of the server to connect to.
            kind: The type of client to create.
            **kwargs: Additional keyword arguments to pass to the client constructor.
        """
        if kind is None or kind == "auto":
            # NOTE: Exhaustive check for all possible client types.
            # TODO: Wrt exceptions, should be using PEP654
            if asyncio.get_event_loop().is_running():
                # We are inside a running event loop, create an async client.
                try:
                    from .http import AsyncHTTPClient

                    return AsyncHTTPClient.from_url(server_url, **kwargs)
                except Exception:
                    try:
                        from .grpc import AsyncGrpcClient

                        return AsyncGrpcClient.from_url(server_url, **kwargs)
                    except Exception as err:
                        logger.error(
                            "Failed to instantiate a client after trying all possible async type. Exception:\n"
                        )
                        logger.error(err)
                        raise BentoMLException(
                            f"Failed to create a BentoML client from given URL '{server_url}'."
                        ) from err
            else:
                try:
                    from .http import HTTPClient

                    return HTTPClient.from_url(server_url, **kwargs)
                except Exception:
                    try:
                        from .grpc import GrpcClient

                        return GrpcClient.from_url(server_url, **kwargs)
                    except Exception as err:
                        logger.error(
                            "Failed to instantiate a client after trying all possible async type. Exception:\n"
                        )
                        logger.error(err)
                        raise BentoMLException(
                            f"Failed to create a BentoML client from given URL '{server_url}': {err} ({err.__class__.__name__})"
                        ) from err
        elif kind == "http":
            from .http import HTTPClient

            return HTTPClient.from_url(server_url, **kwargs)
        elif kind.replace("-", "_") == "async_http":
            from .http import AsyncHTTPClient

            return AsyncHTTPClient.from_url(server_url, **kwargs)
        elif kind.replace("-", "_") == "grpc":
            from .grpc import GrpcClient

            return GrpcClient.from_url(server_url, **kwargs)
        elif kind.replace("-", "_") == "async_grpc":
            from .grpc import AsyncGrpcClient

            return AsyncGrpcClient.from_url(server_url, **kwargs)
        else:
            raise BentoMLException(
                f"Invalid client kind '{kind}'. Must be one of 'http', 'grpc', 'async_http', 'async_grpc' or 'auto'."
            )

    @abstractmethod
    async def _call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        raise NotImplementedError

    supports_kwds_assignment: bool = False
    _endpoint_kwds_map: dict[str, list[str]] | None = None

    def _getitem_from_kwds_map(self, api: InferenceAPI) -> list[str]:
        # type-safe getters
        if self.supports_kwds_assignment:
            assert self._endpoint_kwds_map is not None
            if api.name not in self._endpoint_kwds_map:
                raise BentoMLException(
                    f"{api.name} is not available in OpenAPI spec (malformed spec)."
                )
            return self._endpoint_kwds_map[api.name]
        raise ValueError("Given API Service doesn't support keyword assignment.")

    def _prepare_call_inputs(
        self,
        inp: t.Any | None = None,
        *,
        io_kwargs: dict[str, t.Any],
        api: InferenceAPI,
    ) -> t.Any:
        """Prepare the actual inputs for the call method. With newer service,
        it will also send the function signatures through OpenAPI, hence this will help
        to determine whether the input can be get from the kwargs.

        This function also handles cases for multipart inputs vs. single inputs.
        """
        if api.multi_input:
            if inp is not None:
                raise BentoMLException(
                    f"'{api.name}' takes multiple inputs; all inputs must be passed as keyword arguments."
                )
            if self.supports_kwds_assignment:
                diff = set(io_kwargs).difference(self._getitem_from_kwds_map(api))
                if len(diff) > 0:
                    raise BentoMLException(
                        f"Mismatch input kwargs from spec and given: {diff}"
                    )
            inp = io_kwargs
        else:
            # NOTE: Currently, we only support one input per service APIs.
            attr = self._getitem_from_kwds_map(api)[0]
            if inp is None:
                if attr not in io_kwargs:
                    raise ValueError(
                        f"'inp' is not set, and '{attr}' kwargs is missing."
                    )
                inp = io_kwargs[attr]
        return inp


class BaseSyncClient(Client, client_type="sync"):
    def __init__(self, svc: Service, server_url: str):
        super().__init__(svc, server_url)
        for name, api in self._svc.apis.items():
            if not hasattr(self, name):
                setattr(
                    self,
                    name,
                    wraps_call_attributes(
                        functools.partial(self._sync_call, _bentoml_api=api),
                        api,
                        svc,
                        _func_name=name,
                    ),
                )

            # Backwards compatibility
            if not hasattr(self, f"async_{name}"):
                setattr(
                    self,
                    f"async_{name}",
                    wraps_call_attributes(
                        functools.partial(self._call, _bentoml_api=api),
                        api,
                        svc,
                        _func_name=f"async_{name}",
                        _async_doc=True,
                    ),
                )

    # XXX: This is for backward compatibility only.
    async def async_call(
        self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        assert self._CLIENT_TYPE == ClientType.SYNC
        logger.warning(
            "Calling 'async_call' for '%s' from a sync client is deprecated. Use 'Async%s.call' instead",
            bentoml_api_name,
            self.__class__.__name__,
        )
        return await self._call(
            inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs
        )

    def call(self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any):
        return self._sync_call(
            inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._conn_type:
            try:
                self._conn_type.close()
            except Exception as err:
                logger.error(
                    "Exception while closing client session: %s", self._conn_type
                )
                logger.error(err)
                raise

    def _sync_call(
        self, inp: t.Any = None, *, _bentoml_api: InferenceAPI, **kwargs: t.Any
    ) -> t.Any:
        raise NotImplementedError

    def health(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        raise NotImplementedError("'health' is not implemented.")

    def __del__(self):
        # Close connection when this object is destroyed
        if self._conn_type and hasattr(self._conn_type, "close"):
            try:
                self._conn_type.close()
            except Exception as e:
                logger.error("Exception caught while gc the client object:\n")
                logger.error(e)
                raise


class BaseAsyncClient(Client, client_type="async"):
    def __init__(self, svc: Service, server_url: str):
        super().__init__(svc, server_url)

        for name, api in self._svc.apis.items():
            if not hasattr(self, name):
                setattr(
                    self,
                    name,
                    wraps_call_attributes(
                        functools.partial(self._call, _bentoml_api=api),
                        api,
                        svc,
                        _func_name=name,
                        _async_doc=True,
                    ),
                )

    async def health(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        raise NotImplementedError("'health' is not implemented.")

    async def call(
        self, bentoml_api_name: str, inp: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        return await self._call(
            inp=inp, _bentoml_api=self._svc.apis[bentoml_api_name], **kwargs
        )

    # NOTE: Context-manager related to manage long-running session.
    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._conn_type:
            try:
                await self._conn_type.close()
            except Exception as err:
                logger.error(
                    "Exception while closing client session: %s", self._conn_type
                )
                logger.error(err)
                raise

    def __del__(self):
        # Close connection when this object is destroyed
        if (
            self._conn_type
            and hasattr(self._conn_type, "close")
            and asyncio.iscoroutinefunction(self._conn_type.close)
        ):
            try:
                # We probably want to use `run` here to close the session.
                asyncio.run(self._conn_type.close())
            except Exception as err:
                logger.error("\nException caught while gc the client object:")
                logger.error(err)
                raise
                # pass
