from __future__ import annotations

import typing as t

from bentoml_io.api import APIMethod

if t.TYPE_CHECKING:
    from .client.base import AbstractClient


class Servable:
    __servable_methods__: dict[str, APIMethod[..., t.Any]] = {}
    # User defined attributes
    name: str
    SUPPORTED_RESOURCES: tuple[str, ...] = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING: bool = False

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "name"):
            cls.name = cls.__name__
        new_servable_methods: dict[str, APIMethod[..., t.Any]] = {}
        for attr in vars(cls).values():
            if isinstance(attr, APIMethod):
                new_servable_methods[attr.name] = attr  # type: ignore
        cls.__servable_methods__ = {**cls.__servable_methods__, **new_servable_methods}

    def sync_client(self, name_or_class: str | type[Servable]) -> AbstractClient:
        """Get a synchroneous client for this servable"""
        # To be injected by service
        raise NotImplementedError

    def async_client(self, name_or_class: str | type[Servable]) -> AbstractClient:
        """Get an asynchroneous client for this servable"""
        # To be injected by service
        raise NotImplementedError

    @classmethod
    def schema(cls) -> dict[str, t.Any]:
        return {
            "name": cls.name,
            "type": "service",
            "routes": [method.schema() for method in cls.__servable_methods__.values()],
        }

    def call(self, method_name: str, input_data: dict[str, t.Any]) -> t.Any:
        method = self.__servable_methods__.get(method_name)
        if method is None:
            raise ValueError(f"Method {method_name} not found")
        input_model = method.input_spec(**input_data)
        args = {k: getattr(input_model, k) for k in input_model.model_fields}
        return method.func(self, **args)
