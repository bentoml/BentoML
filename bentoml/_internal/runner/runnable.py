from __future__ import annotations

import typing as t
import logging
from abc import ABCMeta
from typing import overload
from typing import TYPE_CHECKING
from collections.abc import Set

import attr

from ..types import LazyType
from ..types import ParamSpec

if TYPE_CHECKING:
    from ..types import AnyType

T = t.TypeVar("T", bound="RunnableMeta")
P = ParamSpec("P")
R = t.TypeVar("R")

logger = logging.getLogger(__name__)

RUNNABLE_METHOD_MARK: str = "_bentoml_runnable_method"


class RunnableMeta(ABCMeta, t.Type[t.Any]):
    supported_resources: Set[str]
    supports_multi_threading: bool

    methods: dict[str, RunnableMethod[t.Any, t.Any, t.Any]] | None = None

    def __new__(
        cls,
        name: str,
        bases: t.Tuple[type, ...],
        attr_dict: dict[t.Any, t.Any],
        *,
        supported_resources: Set[str] | None = None,
        supports_multi_threading: bool | None = None,
        **_kwargs: t.Any,
    ) -> RunnableMeta:
        res = super().__new__(cls, name, bases, attr_dict)

        if "SUPPORT_NVIDIA_GPU" in attr_dict:
            if supported_resources is None:
                if attr_dict["SUPPORT_NVIDIA_GPU"]:
                    supported_resources = {"nvidia.com/gpu"}
                else:
                    supported_resources = set()
                logger.warning(
                    f"{name} is using deprecated 'SUPPORT_NVIDIA_GPU'. Please convert to using 'supported_resources':\n"
                    f"class {name}(Runnable, supported_resources={supported_resources}):\n"
                    "    ..."
                )
            else:
                logger.warning(
                    f"Deprecated 'SUPPORT_NVIDIA_GPU' is being ignored in favor of 'supported_resources' for {name}."
                )

        if supported_resources is None:
            # attempt to get supported_resources from a superclass
            found_base = None
            for base in bases:
                if isinstance(base, RunnableMeta):
                    if (
                        supported_resources is not None
                        and base.supported_resources != supported_resources
                    ):
                        # we've already set supported_resources
                        raise TypeError(
                            f"Base classes for {name} '{found_base}' and '{base}' have conflicting values for 'supported_resources' ({supported_resources} and {base.supported_resources}, respectively). Please specify 'supported_resources' manually."
                        )

                    found_base = base
                    supported_resources = base.supported_resources

            if supported_resources is None:
                supported_resources = set()

        res.supported_resources = supported_resources

        if "SUPPORT_CPU_MULTI_THREADING" in attr_dict:
            if supports_multi_threading is None:
                logger.warning(
                    f"{name} is using deprecated 'SUPPORT_CPU_MULTI_THREADING'. Please convert to using 'supports_multi_threading':\n"
                    f"class {name}(Runnable, supports_multi_threading=True):\n"
                    "    ..."
                )
                supports_multi_threading = attr_dict["SUPPORT_CPU_MULTI_THREADING"]
            elif attr_dict["SUPPORT_CPU_MULTI_THREADING"] != supports_multi_threading:
                logger.warning(
                    f"Deprecated 'SUPPORT_CPU_MULTI_THREADING' is being ignored in favor of 'supports_multi_threading' for {name}"
                )

        if supports_multi_threading is None:
            # attempt to get supports_multi_threading from a superclass
            found_base = None
            for base in bases:
                if isinstance(base, RunnableMeta):
                    if (
                        supports_multi_threading is not None
                        and base.supports_multi_threading != supports_multi_threading
                    ):
                        # we've already set supported_resources
                        raise TypeError(
                            f"Base classes for {name} '{found_base}' and '{base}' have conflicting values for 'supports_multi_threading' ({supports_multi_threading} and {base.supports_multi_threading}, respectively). Please specify 'supports_multi_threading' manually."
                        )

                    found_base = base
                    supports_multi_threading = base.supports_multi_threading

            if supports_multi_threading is None:
                supports_multi_threading = False

        res.supports_multi_threading = supports_multi_threading

        return res

    def add_method(
        self: RunnableMeta,
        method: t.Callable[t.Concatenate[T, P], t.Any],
        name: str,
        *,
        batchable: bool = False,
        batch_dim: tuple[int, int] | int = 0,
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: LazyType[t.Any] | None = None,
    ):
        meth: RunnableMethod[T, P, t.Any] = Runnable.method(
            method,
            batchable=batchable,
            batch_dim=batch_dim,
            input_spec=input_spec,
            output_spec=output_spec,
        )
        setattr(self, name, meth)
        meth.__set_name__(self, name)

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


class Runnable(metaclass=RunnableMeta):
    pass


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
        if owner.methods is None:
            owner.methods = {}
        owner.methods[name] = self


@attr.define
class RunnableMethodConfig:
    batchable: bool
    batch_dim: tuple[int, int]
    input_spec: AnyType | t.Tuple[AnyType, ...] | None = None
    output_spec: AnyType | None = None
