from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from functools import update_wrapper

from ...exceptions import InvalidArgument

if t.TYPE_CHECKING:
    from types import UnionType

    import pyarrow
    import pyspark.sql.types
    from starlette.requests import Request
    from starlette.responses import Response

    from bentoml.grpc.types import ProtoField

    from ..types import LazyType
    from ..context import ServiceContext as Context
    from ..service.openapi.specification import Schema
    from ..service.openapi.specification import Reference

    InputType = (
        UnionType
        | t.Type[t.Any]
        | LazyType[t.Any]
        | dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]
    )
    OpenAPIResponse = dict[str, str | dict[str, t.Any]]
    F = t.Callable[..., t.Any]


IO_DESCRIPTOR_REGISTRY: dict[str, type[IODescriptor[t.Any]]] = {}

IOType = t.TypeVar("IOType")


def from_spec(spec: dict[str, t.Any]) -> IODescriptor[t.Any]:
    if "id" not in spec:
        raise InvalidArgument(f"IO descriptor spec ({spec}) missing ID.")

    if spec["id"] is None:
        raise InvalidArgument("No IO descriptor spec found.")

    return IO_DESCRIPTOR_REGISTRY[spec["id"]].from_spec(spec)


class _OpenAPIMeta:
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


class IODescriptor(ABC, _OpenAPIMeta, t.Generic[IOType]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    if t.TYPE_CHECKING:
        # Populated by subclasses. Makes pyright happy.
        def __init__(self, **kwargs: t.Any) -> None:
            ...

    __slots__ = (
        "_mime_type",
        "proto_fields",
        "descriptor_id",
        "from_sample",
        "_sample",
    )

    HTTP_METHODS = ("POST",)

    _mime_type: str
    descriptor_id: str | None
    proto_fields: tuple[ProtoField]

    def __init_subclass__(
        cls,
        *,
        descriptor_id: str | None = None,
        proto_fields: tuple[ProtoField] | None = None,
    ):
        if descriptor_id is not None:
            if descriptor_id in IO_DESCRIPTOR_REGISTRY:
                raise ValueError(
                    f"Descriptor ID {descriptor_id} already registered to {IO_DESCRIPTOR_REGISTRY[descriptor_id]}."
                )
            IO_DESCRIPTOR_REGISTRY[descriptor_id] = cls
        cls.descriptor_id = descriptor_id
        cls.proto_fields = proto_fields or t.cast("tuple[ProtoField]", ())

        cls._sample: IOType | None = None

        def _() -> F:
            def impl(
                cl_: type[IODescriptor[t.Any]], sample: t.Any, **kwargs: t.Any
            ) -> IODescriptor[t.Any]:
                klass = cl_(**kwargs)
                klass.sample = klass._from_sample(sample)
                return klass

            impl.__doc__ = cls._from_sample.__doc__
            return update_wrapper(impl, cls._from_sample)

        cls.from_sample = classmethod(_())

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @property
    def sample(self) -> IOType | None:
        return self._sample

    @sample.setter
    def sample(self, value: IOType) -> None:
        self._sample = value

    @abstractmethod
    def _from_sample(self, sample: t.Any) -> IOType:
        """
        Creates a new instance of the IO Descriptor from given sample.

        Args:
            sample: The sample to create the instance from.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            An instance of the IODescriptor.
        """
        raise NotImplementedError

    @property
    def mime_type(self) -> str:
        return self._mime_type

    def to_spec(self) -> dict[str, t.Any] | None:
        return None

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        raise NotImplementedError

    @abstractmethod
    def input_type(self) -> InputType:
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
    async def from_proto(self, field: t.Any) -> IOType:
        raise NotImplementedError

    @abstractmethod
    async def to_proto(self, obj: IOType) -> t.Any:
        raise NotImplementedError

    def from_arrow(self, batch: pyarrow.RecordBatch) -> IOType:
        raise NotImplementedError(
            "This IO descriptor does not currently support batch inference."
        )

    def to_arrow(self, obj: IOType) -> pyarrow.RecordBatch:
        raise NotImplementedError(
            "This IO descriptor does not currently support batch inference."
        )

    def spark_schema(self) -> pyspark.sql.types.StructType:
        raise NotImplementedError(
            "This IO descriptor does not currently support batch inference."
        )
