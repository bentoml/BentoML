from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from ...exceptions import InvalidArgument

if TYPE_CHECKING:
    from types import UnionType

    from typing_extensions import Self
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
        | t.Type[t.Any]
        | LazyType[t.Any]
        | dict[str, t.Type[t.Any] | UnionType | LazyType[t.Any]]
    )
    OpenAPIResponse = dict[str, str | dict[str, MediaType] | dict[str, t.Any]]


IO_DESCRIPTOR_REGISTRY: dict[str, type[IODescriptor[t.Any]]] = {}

IOType = t.TypeVar("IOType")


def from_spec(spec: dict[str, str]) -> IODescriptor[t.Any]:
    if "id" not in spec:
        raise InvalidArgument(f"IO descriptor spec ({spec}) missing ID.")
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

    __slots__ = ("_args", "_kwargs", "_proto_fields", "_mime_type", "descriptor_id")

    HTTP_METHODS = ["POST"]

    descriptor_id: str | None

    _mime_type: str
    _rpc_content_type: str = "application/grpc"
    _proto_fields: tuple[ProtoField]
    _sample: IOType | None = None
    _args: t.Sequence[t.Any]
    _kwargs: dict[str, t.Any]

    def __init_subclass__(cls, *, descriptor_id: str | None = None):
        if descriptor_id is not None:
            if descriptor_id in IO_DESCRIPTOR_REGISTRY:
                raise ValueError(
                    f"Descriptor ID {descriptor_id} already registered to {IO_DESCRIPTOR_REGISTRY[descriptor_id]}."
                )
            IO_DESCRIPTOR_REGISTRY[descriptor_id] = cls
        cls.descriptor_id = descriptor_id

    if TYPE_CHECKING:

        def __init__(self, **kwargs: t.Any) -> None:
            ...

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @property
    def sample(self) -> IOType | None:
        return self._sample

    @sample.setter
    def sample(self, value: IOType) -> None:
        self._sample = value

    @classmethod
    def from_sample(cls, sample: IOType | t.Any, **kwargs: t.Any) -> Self:
        klass = cls(**kwargs)
        klass.sample = klass._from_sample(sample)
        return klass

    @abstractmethod
    def _from_sample(self, sample: t.Any) -> IOType:
        raise NotImplementedError

    @property
    def mime_type(self) -> str:
        return self._mime_type

    @abstractmethod
    def to_spec(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> Self:
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
