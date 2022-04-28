from __future__ import annotations

import typing as t
import inspect
import logging
import functools
from abc import ABC
from abc import abstractmethod
from typing import overload
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    WrappedMethod = t.TypeVar("WrappedMethod", bound=t.Callable[..., t.Any])
    BatchDimType: t.TypeAlias = t.Tuple[t.List[int] | int, t.List[int] | int] | int

import attr

from bentoml._internal.types import LazyType

logger = logging.getLogger(__name__)

RUNNABLE_METHOD_MARK: str = "_bentoml_runnable_method"


def method_decorator(
    meth: WrappedMethod,
    batchable: bool = False,
    batch_dim: BatchDimType = 0,
    input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
    output_spec: LazyType[t.Any] | None = None,
) -> WrappedMethod:
    setattr(
        meth,
        RUNNABLE_METHOD_MARK,
        RunnableMethodConfig(
            batchable=batchable,
            batch_dim=batch_dim,
            input_spec=input_spec,
            output_spec=output_spec,
        ),
    )
    return meth


class Runnable(ABC):
    @property
    @abstractmethod
    def supports_nvidia_gpu(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_multi_threading(self) -> bool:
        ...

    @overload
    @staticmethod
    def method(
        batchable_or_method: WrappedMethod,
        batch_dim: BatchDimType = 0,
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: LazyType[t.Any] | None = None,
    ) -> WrappedMethod:
        ...

    @overload
    @staticmethod
    def method(
        batchable_or_method: bool,
        batch_dim: BatchDimType = 0,
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: LazyType[t.Any] | None = None,
    ) -> t.Callable[[WrappedMethod], WrappedMethod]:
        ...

    @staticmethod
    def method(
        batchable_or_method: bool | WrappedMethod = False,  # type: ignore (pyright bug?)
        batch_dim: BatchDimType = 0,
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: t.Optional[LazyType[t.Any]] = None,
    ) -> t.Callable[[WrappedMethod], WrappedMethod] | WrappedMethod:
        if callable(batchable_or_method):
            return method_decorator(
                batchable_or_method,
                batch_dim=batch_dim,
                batchable=True,
                input_spec=input_spec,
                output_spec=output_spec,
            )
        return functools.partial(
            method_decorator,
            batchable=batchable_or_method,
            batch_dim=batch_dim,
            input_spec=input_spec,
            output_spec=output_spec,
        )

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_method_configs(cls) -> t.Dict[str, RunnableMethodConfig]:
        return {
            name: getattr(meth, RUNNABLE_METHOD_MARK)
            for name, meth in inspect.getmembers(
                cls, predicate=lambda x: hasattr(x, RUNNABLE_METHOD_MARK)
            )
        }


@attr.define()
class RunnableMethodConfig:
    batchable: bool = attr.field()
    batch_dim: BatchDimType = attr.field()
    input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = attr.field(
        default=None
    )
    output_spec: LazyType[t.Any] | None = attr.field(default=None)
