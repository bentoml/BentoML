from __future__ import annotations

import typing as t
import inspect
import logging
import functools
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    WrappedMethod = t.TypeVar("WrappedMethod", bound=t.Callable[..., t.Any])
    BatchDimType: t.TypeAlias = t.Tuple[t.List[int] | int, t.List[int] | int] | int

import attr

from bentoml._internal.types import AnyType
from bentoml._internal.types import LazyType

logger = logging.getLogger(__name__)

RUNNABLE_METHOD_MARK: str = "_bentoml_runnable_method"


class Runnable(ABC):
    SUPPORT_NVIDIA_GPU: bool
    SUPPORT_CPU_MULTI_THREADING: bool

    @classmethod
    def add_method(
        cls,
        method: t.Callable[..., t.Any],
        name: str,
        *,
        batchable: bool = False,
        batch_dim: BatchDimType = 0,
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: LazyType[t.Any] | None = None,
    ):
        setattr(
            cls,
            name,
            Runnable.method(
                method,
                batchable=batchable,
                batch_dim=batch_dim,
                input_spec=input_spec,
                output_spec=output_spec,
            ),
        )

    @staticmethod
    def method(
        meth: WrappedMethod | None = None,
        *,
        batchable: bool = False,
        batch_dim: BatchDimType = 0,
        input_spec: AnyType | tuple[AnyType, ...] | None = None,
        output_spec: AnyType | None = None,
    ) -> t.Callable[[WrappedMethod], WrappedMethod] | WrappedMethod:
        def method_decorator(meth: WrappedMethod) -> WrappedMethod:
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

        if callable(meth):
            return method_decorator(meth)
        return method_decorator

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
    batchable: bool
    batch_dim: BatchDimType
    input_spec: AnyType | t.Tuple[AnyType, ...] | None = None
    output_spec: AnyType | None = None
