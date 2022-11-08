from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from ..utils import singledispatchmethod
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


@singledispatchmethod
def create_sample(self: IODescriptor[t.Any], value: t.Any) -> None:
    raise InvalidArgument(
        f"Unsupported sample type: '{type(value)}' (value: {value}). To register type '{type(value)}' to {self.__class__.__name__} implement a dispatch function and register types to 'create_sample.register'"
    )


def from_spec(spec: dict[str, str]) -> IODescriptor[t.Any]:
    if "id" not in spec:
        raise InvalidArgument(f"IO descriptor spec ({spec}) missing ID.")
    return IO_DESCRIPTOR_REGISTRY[spec["id"]].from_spec(spec)


class IODescriptor(ABC, t.Generic[IOType]):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    HTTP_METHODS = ["POST"]

    descriptor_id: str | None

    _mime_type: str
    _rpc_content_type: str = "application/grpc"
    _proto_fields: tuple[ProtoField]
    _sample: IOType | None = None
    _create_sample: singledispatchmethod[None] = create_sample

    def __init_subclass__(cls, *, descriptor_id: str | None = None):
        if descriptor_id is not None:
            if descriptor_id in IO_DESCRIPTOR_REGISTRY:
                raise ValueError(
                    f"Descriptor ID {descriptor_id} already registered to {IO_DESCRIPTOR_REGISTRY[descriptor_id]}."
                )
            IO_DESCRIPTOR_REGISTRY[descriptor_id] = cls
        cls.descriptor_id = descriptor_id

    def __new__(cls, *args: t.Any, **kwargs: t.Any):
        sample = kwargs.pop("_sample", None)
        kls = super().__new__(cls)
        if sample is not None:
            kls._create_sample(sample)
        return kls

    @property
    def sample(self) -> IOType | None:
        return self._sample

    @sample.setter
    def sample(self, value: IOType) -> None:
        self._sample = value

    # NOTE: for custom types handle, use 'create_sample.register' to register
    # custom types for 'from_sample'
    @classmethod
    @abstractmethod
    def from_sample(cls, sample: IOType | t.Any, **kwargs: t.Any) -> Self:
        return cls.__new__(cls, _sample=sample, **kwargs)

    @abstractmethod
    def to_spec(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> Self:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @abstractmethod
    def input_type(self) -> InputType:
        raise NotImplementedError

    @abstractmethod
    def openapi_schema(self) -> Schema | Reference:
        raise NotImplementedError

    def openapi_example(self) -> t.Any:
        if self.sample is not None:
            return self.sample

    @abstractmethod
    def openapi_components(self) -> dict[str, t.Any] | None:
        raise NotImplementedError

    @abstractmethod
    def openapi_request_body(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @abstractmethod
    def openapi_responses(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @abstractmethod
    async def from_http_request(self, request: Request) -> IOType:
        ...

    @abstractmethod
    async def to_http_response(
        self, obj: IOType, ctx: Context | None = None
    ) -> Response:
        ...

    @abstractmethod
    async def from_proto(self, field: t.Any) -> IOType:
        ...

    @abstractmethod
    async def to_proto(self, obj: IOType) -> t.Any:
        ...
