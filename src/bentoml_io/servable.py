from __future__ import annotations

import typing as t

from bentoml_io.api import APIMethod

if t.TYPE_CHECKING:
    from .client.base import AbstractClient


class Servable:
    __servable_methods__: dict[str, APIMethod[..., t.Any]] = {}
    __clients_cache: dict[str, AbstractClient]
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
                new_servable_methods[attr.name] = attr
        cls.__servable_methods__ = {**cls.__servable_methods__, **new_servable_methods}

    def get_client(self, name_or_class: str | type[Servable]) -> AbstractClient:
        if not hasattr(self, "__clients_cache"):
            self.__clients_cache = {}
        name = name_or_class if isinstance(name_or_class, str) else name_or_class.name
        if name not in self.__clients_cache:
            # TODO: create client
            ...
        return self.__clients_cache[name]

    def schema(self) -> dict[str, t.Any]:
        return {
            "name": self.__class__.__name__,
            "type": "service",
            "routes": [
                method.schema() for method in self.__servable_methods__.values()
            ],
        }

    def call(self, method_name: str, input_data: dict[str, t.Any]) -> t.Any:
        method = self.__servable_methods__.get(method_name)
        if method is None:
            raise ValueError(f"Method {method_name} not found")
        input_model = method.input_spec(**input_data)
        args = {k: getattr(input_model, k) for k in input_model.model_fields}
        return method.func(self, **args)
