from __future__ import annotations

import abc
import typing as t

import attrs

from _bentoml_sdk import IODescriptor

T = t.TypeVar("T")


@attrs.define(slots=True)
class ClientEndpoint:
    name: str
    route: str
    doc: str | None = None
    input: dict[str, t.Any] = attrs.field(factory=dict)
    output: dict[str, t.Any] = attrs.field(factory=dict)
    input_spec: type[IODescriptor] | None = None
    output_spec: type[IODescriptor] | None = None
    stream_output: bool = False


class AbstractClient(abc.ABC):
    endpoints: dict[str, ClientEndpoint]

    def __init__(self) -> None:
        for name in self.endpoints:
            attr_name = name
            if getattr(self, attr_name, None) is not None:
                attr_name = f"api_{name}"  # prefix to avoid name conflict
            setattr(self, attr_name, self._make_method(name))

    def _make_method(self, name: str) -> t.Callable[..., t.Any]:
        endpoint = self.endpoints[name]

        def method(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return self.call(name, *args, **kwargs)

        method.__doc__ = endpoint.doc
        if endpoint.input_spec is not None:
            method.__annotations__ = endpoint.input_spec.__annotations__
            method.__signature__ = endpoint.input_spec.__signature__
        return method

    @abc.abstractmethod
    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Call a service method by its name.
        It takes the same arguments as the service method.
        """
