from __future__ import annotations

import functools
import inspect
import logging
import typing as t
from typing import TYPE_CHECKING
from typing import overload

import attr

from ...exceptions import BentoMLException
from ..ionext.function import ensure_io_descriptor
from ..ionext.function import get_input_spec
from ..ionext.function import get_output_spec

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..ionext.models import IODescriptor

    # only use ParamSpec in type checking, as it's only in 3.10
    P = t.ParamSpec("P")
else:
    P = t.TypeVar("P")

T = t.TypeVar("T", bound="Runnable")
R = t.TypeVar("R")

logger = logging.getLogger(__name__)

RUNNABLE_METHOD_MARK: str = "_bentoml_runnable_method"


class Runnable:
    SUPPORTED_RESOURCES: tuple[str, ...]
    SUPPORTS_CPU_MULTI_THREADING: bool

    bentoml_runnable_methods__: dict[str, RunnableMethod[t.Any, t.Any, t.Any]] = {}

    def __setattr__(self, attr_name: str, value: t.Any):
        if attr_name in ("SUPPORTED_RESOURCES", "SUPPORTS_CPU_MULTI_THREADING"):
            # TODO: add link to custom runner documentation
            raise BentoMLException(
                f"{attr_name} should not be set at runtime; the change will not be reflected in the scheduling strategy. Instead, create separate Runnables with different supported resource configurations."
            )

        super().__setattr__(attr_name, value)

    def __getattribute__(self, item: str) -> t.Any:
        if item in ["add_method", "method"]:
            # TODO: add link to custom runner documentation
            raise BentoMLException(
                f"{item} should not be used at runtime; instead, use {type(self).__name__}.{item} where you define the class."
            )

        return super().__getattribute__(item)

    @classmethod
    def add_method(
        cls: t.Type[T],
        method: t.Callable[t.Concatenate[T, P], t.Any],
        name: str,
        *,
        batchable: bool = False,
        batch_dim: tuple[int, int] | int = 0,
        input_spec: type[BaseModel] | None = None,
        output_spec: type | None = None,
    ):
        meth = Runnable.method(
            method,
            batchable=batchable,
            batch_dim=batch_dim,
            input_spec=input_spec,
            output_spec=output_spec,
        )
        setattr(cls, name, meth)
        meth.__set_name__(cls, name)

    @overload
    @staticmethod
    def method(
        meth: t.Callable[t.Concatenate[T, P], R],
        *,
        batchable: bool = False,
        batch_dim: tuple[int, int] | int = 0,
        input_spec: type[BaseModel] | None = None,
        output_spec: type | None = None,
    ) -> RunnableMethod[T, P, R]:
        ...

    @overload
    @staticmethod
    def method(
        meth: None = None,
        *,
        batchable: bool = False,
        batch_dim: tuple[int, int] | int = 0,
        input_spec: type[BaseModel] | None = None,
        output_spec: type | None = None,
    ) -> t.Callable[[t.Callable[t.Concatenate[T, P], R]], RunnableMethod[T, P, R]]:
        ...

    @staticmethod
    def method(
        meth: t.Callable[t.Concatenate[T, P], R] | None = None,
        *,
        batchable: bool = False,
        batch_dim: tuple[int, int] | int = 0,
        input_spec: type[BaseModel] | None = None,
        output_spec: type | None = None,
    ) -> (
        t.Callable[[t.Callable[t.Concatenate[T, P], R]], RunnableMethod[T, P, R]]
        | RunnableMethod[T, P, R]
    ):
        def method_decorator(
            meth: t.Callable[t.Concatenate[T, P], R]
        ) -> RunnableMethod[T, P, R]:
            nonlocal input_spec, output_spec
            if input_spec is None:
                input_spec = get_input_spec(meth, skip_self=True)
            else:
                input_spec = ensure_io_descriptor(input_spec)
            if output_spec is None:
                output_spec = get_output_spec(meth)
            else:
                output_spec = ensure_io_descriptor(output_spec)
            return RunnableMethod(
                meth,
                RunnableMethodConfig(
                    is_stream=inspect.isasyncgenfunction(meth)
                    or inspect.isgeneratorfunction(meth),
                    batchable=batchable,
                    batch_dim=(batch_dim, batch_dim)
                    if isinstance(batch_dim, int)
                    else batch_dim,
                    input_spec=input_spec,
                    output_spec=output_spec,
                ),
            )

        if callable(meth):
            return method_decorator(meth)
        return method_decorator


@attr.define
class RunnableMethod(t.Generic[T, P, R]):
    func: t.Callable[t.Concatenate[T, P], R]
    config: RunnableMethodConfig
    _bentoml_runnable_method: None = None

    def __get__(self, obj: T, _: t.Type[T] | None = None) -> t.Callable[P, R]:
        def method(*args: P.args, **kwargs: P.kwargs) -> R:
            return self.func(obj, *args, **kwargs)

        signature = inspect.signature(self.func)
        # skip self
        self_arg, *new_params = signature.parameters.values()
        method = functools.update_wrapper(method, self.func)
        method.__signature__ = signature.replace(parameters=new_params)
        method.__annotations__.pop(self_arg.name, None)
        return method

    def __set_name__(self, owner: t.Any, name: str):
        owner.bentoml_runnable_methods__[name] = self


@attr.define
class RunnableMethodConfig:
    batchable: bool
    batch_dim: tuple[int, int]
    input_spec: type[IODescriptor] | None
    output_spec: type[IODescriptor] | None
    is_stream: bool = False
