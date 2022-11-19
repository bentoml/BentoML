from __future__ import annotations

import typing as t
import asyncio
import inspect
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from functools import update_wrapper

import anyio

from .dispatch import MultiStrategyDispatch
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ...grpc.utils import LATEST_STUB_VERSION

if TYPE_CHECKING:
    from types import UnionType

    from google.protobuf import message as _message
    from starlette.requests import Request
    from starlette.responses import Response

    from bentoml.grpc.types import ProtoField

    from ..types import LazyType
    from ..context import InferenceApiContext as Context
    from ..service.openapi.specification import Schema
    from ..service.openapi.specification import MediaType
    from ..service.openapi.specification import Reference

    InputType = (
        UnionType
        | type[t.Any]
        | LazyType[t.Any]
        | dict[str, type[t.Any] | UnionType | LazyType[t.Any]]
    )
    OpenAPIResponse = dict[str, str | dict[str, MediaType] | dict[str, t.Any]]
    # Sync with generated stubs version.
    StubVersion = t.Literal["v1", "v1alpha1"]
    F = t.Callable[..., t.Any]


IO_DESCRIPTOR_REGISTRY: dict[str, type[IODescriptor[t.Any]]] = {}

T = t.TypeVar("T")
IOType = t.TypeVar("IOType")


class IOStructureError(BentoMLException):
    """
    Raise when structure function failed to convert the input data to the desired type
    """


def from_spec(spec: dict[str, str]) -> IODescriptor[t.Any]:
    if "id" not in spec:
        raise InvalidArgument(f"IO descriptor spec ({spec}) missing ID.")
    return IO_DESCRIPTOR_REGISTRY[spec["id"]].from_spec(spec)


class OpenAPIMixin:
    @abstractmethod
    def openapi_schema(self) -> Schema | Reference:
        raise NotImplementedError

    @abstractmethod
    def openapi_components(self) -> dict[str, t.Any] | None:
        raise NotImplementedError

    @abstractmethod
    def openapi_example(self) -> t.Any | None:
        raise NotImplementedError

    @abstractmethod
    def openapi_request_body(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @abstractmethod
    def openapi_responses(self) -> dict[str, t.Any]:
        raise NotImplementedError


DEFAULT_FROM_SAMPLE_DOCSTRING = """\
Creates an instance of the :class:`{name}` from given sample. Note that ``from_sample`` does not
take positional arguments.

Args:
    sample: The sample to create the instance from.
    kwargs: Additional keyword arguments to pass to the constructor.

Returns:
    :class:`{name}`: The instance created from the sample.
"""


class IODescriptor(ABC, t.Generic[IOType], OpenAPIMixin):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    if TYPE_CHECKING:

        # Populated by subclasses. Makes IDE happy.
        def __init__(self, **kwargs: t.Any) -> None:
            ...

    __slots__ = (
        "_mime_type",
        "proto_fields",
        "descriptor_id",
        "from_sample",
        "HTTP_METHODS",
        "_sample",
        "_unstructure_proto_fn",
        "_structure_proto_fn",
        "_struct_fn_registered",
        "_unstruct_fn_registered",
    )

    _mime_type: str
    descriptor_id: str | None
    proto_fields: tuple[ProtoField]

    def __init_subclass__(
        cls,
        *,
        descriptor_id: str | None = None,
        proto_fields: tuple[ProtoField] | None = None,
    ) -> None:
        cls.HTTP_METHODS: tuple[str] = ("POST",)

        if descriptor_id is not None:
            if descriptor_id in IO_DESCRIPTOR_REGISTRY:
                raise ValueError(
                    f"Descriptor ID {descriptor_id} already registered to {IO_DESCRIPTOR_REGISTRY[descriptor_id]}."
                )
            IO_DESCRIPTOR_REGISTRY.setdefault(descriptor_id, cls)
        cls.descriptor_id = descriptor_id

        if proto_fields is not None:
            cls.proto_fields = proto_fields
        else:
            cls.proto_fields = tuple()

        # _unstructure_proto_fn and _structure_proto_fn handle
        # dispatching proto processor depending on types.
        cls._unstructure_proto_fn = MultiStrategyDispatch(cls._structure_error)
        cls._structure_proto_fn = MultiStrategyDispatch(cls._structure_error)
        # The two below boolean are used to check whether each IO descriptor strategy
        # has been registered or not.
        cls._struct_fn_registered = False
        cls._unstruct_fn_registered = False

        cls._sample: IOType | None = None

        # from_sample implementation with correct docstring.
        from_sample_docstring = inspect.getdoc(cls.preprocess_sample)
        if not from_sample_docstring:
            from_sample_docstring = DEFAULT_FROM_SAMPLE_DOCSTRING.format(
                name=cls.__name__
            )

        def _() -> F:
            def __(
                cl_: type[IODescriptor[t.Any]], sample: t.Any, **kwargs: t.Any
            ) -> IODescriptor[t.Any]:
                klass = cl_(**kwargs)
                object.__setattr__(klass, "_sample", klass.preprocess_sample(sample))
                return klass

            __.__doc__ = from_sample_docstring
            return update_wrapper(__, cls.preprocess_sample)

        cls.from_sample = classmethod(_())

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @staticmethod
    def _structure_error(cl: type[t.Any]) -> t.NoReturn:
        """At the bottom of the loop, explode if we can't find a matching type."""
        raise IOStructureError(
            f"Unsupported type: {cl!r}. Register a custom hooks for it."
        )

    @abstractmethod
    def to_spec(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        raise NotImplementedError

    @abstractmethod
    def input_type(self) -> InputType:
        raise NotImplementedError

    @property
    def mime_type(self) -> str:
        return self._mime_type

    @abstractmethod
    def preprocess_sample(self, sample: t.Any) -> IOType:
        raise NotImplementedError

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOType:
        raise NotImplementedError

    @abstractmethod
    async def to_http_response(
        self, obj: IOType, ctx: Context | None = None
    ) -> Response:
        raise NotImplementedError

    @abstractmethod
    def _register_unstructure_proto_fn(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _register_structure_proto_fn(self) -> None:
        raise NotImplementedError

    async def from_proto(self, field: t.Any) -> IOType:
        if not self._unstruct_fn_registered:
            self._register_unstructure_proto_fn()
            self._unstruct_fn_registered = True
        _ = self._unstructure_proto_fn.dispatch(type(field))
        if asyncio.iscoroutinefunction(_):
            return await _(field)
        return await anyio.to_thread.run_sync(_, field)

    async def to_proto(
        self, obj: IOType | t.Any, *, _version: str = LATEST_STUB_VERSION
    ) -> _message.Message:
        if not self._struct_fn_registered:
            self._register_structure_proto_fn()
            self._struct_fn_registered = True
        func = self._structure_proto_fn.dispatch(type(obj))
        if func == self._structure_error:
            self._structure_error(type(obj))

        # NOTE: the function registered under dispatcher must return a dictionary
        # and takes two arguments, the first one is the object to be converted, the second
        # one is the stub version to be used for conversion.
        sig = inspect.signature(func)
        ret_ann = sig.return_annotation
        if len(sig.parameters) != 2:
            raise ValueError(
                f"Invalid structure function {func.__name__} for 'to_proto', must take 2 arguments. First arg is the object to be converted, the second arg is the stub version to be used for conversion. Got {sig!r} instead."
            )
        if ret_ann == inspect.Signature.empty:
            raise ValueError(
                f"Invalid structure function {func.__name__} for 'to_proto', must have return type annotation."
            )
        if asyncio.iscoroutinefunction(func):
            return await func(obj, _version)
        return await anyio.to_thread.run_sync(func, obj, _version)
