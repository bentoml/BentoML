from __future__ import annotations

import typing as t
import inspect
from abc import abstractmethod
from typing import TYPE_CHECKING
from functools import update_wrapper

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
        | type[t.Any]
        | LazyType[t.Any]
        | dict[str, type[t.Any] | UnionType | LazyType[t.Any]]
    )
    OpenAPIResponse = dict[str, str | dict[str, MediaType] | dict[str, t.Any]]

    F = t.Callable[..., t.Any]


IO_DESCRIPTOR_REGISTRY: dict[str, type[IODescriptor[t.Any]]] = {}

IOType = t.TypeVar("IOType")
T = t.TypeVar("T")


def from_spec(spec: dict[str, str]) -> IODescriptor[t.Any]:
    if "id" not in spec:
        raise InvalidArgument(f"IO descriptor spec ({spec}) missing ID.")
    return IO_DESCRIPTOR_REGISTRY[spec["id"]].from_spec(spec)


DEFAULT_FROM_SAMPLE_DOCSTRING = """\
Creates an instance of {name} from given sample. Note that ``from_sample`` does not
take positional arguments.

Args:
    sample: The sample to create the instance from.
    kwargs: Additional keyword arguments to pass to the constructor.

Returns:
    {name}: The instance created from the sample.
"""


class DescriptorMetaclass(type):
    def __new__(
        mcls: type[type[T]],  # type: ignore
        name: str,
        bases: tuple[type[t.Any], ...],
        attrs: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> type[T]:
        # HTTP_METHODS will be the default attrs for all IODescriptor
        attrs["HTTP_METHODS"] = ("POST",)

        for base in bases:
            # This case is reserved for implementation such as File where we have a `_from_sample`
            # in the base class, and each subsequent IO descriptor that extends File won't
            # have to implement it again.
            # NOTE that we also have to check if given `_from_sample` is an abstract class.
            # This is because we want to avoid the from_sample abstractmethod.
            if "_from_sample" in base.__dict__ and not getattr(
                base.__dict__["_from_sample"], "__isabstractmethod__", False
            ):
                _from_sample_impl = base.__dict__["_from_sample"]
                break
        else:
            # GOTO, this is the case where each subsequent IO descriptor will have to implement
            # its own `_from_sample` method.
            if "_from_sample" not in attrs:
                raise NotImplementedError(f"{name} must implement '_from_sample'.")
            _from_sample_impl = attrs["_from_sample"]

        # By default, the documentation for `from_sample` can be specified under `_from_sample`.
        # We will try to get the docstring from `_from_sample` and patch it to `from_sample`.
        # If there is no docstring, we will use the default docstring.
        from_sample_docstring = inspect.getdoc(_from_sample_impl)
        if not from_sample_docstring:
            from_sample_docstring = DEFAULT_FROM_SAMPLE_DOCSTRING.format(name=name)
        # This is `from_sample` implementation, where we will update the docstring to.
        def _(impl: F) -> F:
            def __from_sample(
                cls: type[IODescriptor[t.Any]], sample: t.Any, **kwargs: t.Any
            ) -> IODescriptor[t.Any]:
                klass = cls(**kwargs)
                klass.sample = impl(klass, sample)
                return klass

            __from_sample.__doc__ = from_sample_docstring
            return update_wrapper(__from_sample, impl)

        # Ensure that `from_sample` is the classmethod of any given IO descriptor.
        if "from_sample" not in attrs:
            attrs["from_sample"] = classmethod(_(_from_sample_impl))

        return super().__new__(mcls, name, bases, attrs, **kwargs)


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


class IODescriptor(t.Generic[IOType], _OpenAPIMeta, metaclass=DescriptorMetaclass):
    """
    IODescriptor describes the input/output data format of an InferenceAPI defined
    in a :code:`bentoml.Service`. This is an abstract base class for extending new HTTP
    endpoint IO descriptor types in BentoServer.
    """

    __slots__ = ("_proto_fields", "_mime_type", "descriptor_id")

    descriptor_id: str | None
    _mime_type: str
    _proto_fields: tuple[ProtoField]

    _sample: IOType | None = None

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

    @property
    def mime_type(self) -> str:
        return self._mime_type

    @mime_type.setter
    def mime_type(self, value: str) -> None:
        self._mime_type = value

    @abstractmethod
    def _from_sample(self, sample: t.Any) -> IOType:
        raise NotImplementedError

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
