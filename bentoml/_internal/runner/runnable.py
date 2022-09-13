from __future__ import annotations

import typing as t
import logging
from typing import overload
from typing import TYPE_CHECKING

import attr

from ..types import LazyType
from ...exceptions import BentoMLException

if TYPE_CHECKING:
    from ..types import AnyType

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

    bentoml_runnable_methods__: dict[
        str, RunnableMethod[t.Any, t.Any, t.Any]
    ] | None = None

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
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: LazyType[t.Any] | None = None,
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
        input_spec: AnyType | tuple[AnyType, ...] | None = None,
        output_spec: AnyType | None = None,
    ) -> RunnableMethod[T, P, R]:
        ...

    @overload
    @staticmethod
    def method(
        meth: None = None,
        *,
        batchable: bool = False,
        batch_dim: tuple[int, int] | int = 0,
        input_spec: AnyType | tuple[AnyType, ...] | None = None,
        output_spec: AnyType | None = None,
    ) -> t.Callable[[t.Callable[t.Concatenate[T, P], R]], RunnableMethod[T, P, R]]:
        ...

    @staticmethod
    def method(
        meth: t.Callable[t.Concatenate[T, P], R] | None = None,
        *,
        batchable: bool = False,
        batch_dim: tuple[int, int] | int = 0,
        input_spec: AnyType | tuple[AnyType, ...] | None = None,
        output_spec: AnyType | None = None,
    ) -> t.Callable[
        [t.Callable[t.Concatenate[T, P], R]], RunnableMethod[T, P, R]
    ] | RunnableMethod[T, P, R]:
        def method_decorator(
            meth: t.Callable[t.Concatenate[T, P], R]
        ) -> RunnableMethod[T, P, R]:
            return RunnableMethod(
                meth,
                RunnableMethodConfig(
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

        return method

    def __set_name__(self, owner: t.Any, name: str):
        if owner.bentoml_runnable_methods__ is None:
            owner.bentoml_runnable_methods__ = {}
        owner.bentoml_runnable_methods__[name] = self


@attr.define
class RunnableMethodConfig:
    batchable: bool
    batch_dim: tuple[int, int]
    input_spec: AnyType | tuple[AnyType, ...] | None = None
    output_spec: AnyType | None = None
