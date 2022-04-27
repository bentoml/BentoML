from __future__ import annotations

import abc
import typing as t
import logging
import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    InputType = t.TypeVar("InputType")
    OutputType = t.TypeVar("OutputType")

    WrappedMethod = t.TypeVar("WrappedMethod", bound=t.Callable[..., t.Any])
    BatchDimType: t.TypeAlias = t.Tuple[t.List[int] | int, t.List[int] | int] | int

import attr

from bentoml._internal.types import LazyType

logger = logging.getLogger(__name__)


class Runnable(abc.ABC):
    SUPPORT_NVIDIA_GPU: bool
    SUPPORT_MULTIPLE_CPU_THREADS: bool

    @staticmethod
    def method(
        batchable_or_method: bool | WrappedMethod = False,
        batch_dim: int = 0,
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: t.Optional[LazyType[t.Any]] = None,
    ) -> t.Callable[[], WrappedMethod] | WrappedMethod:
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
    def get_runnable_methods(cls) -> t.List[RunnableMethodConfig]:
        if cls.__runnable_methods is None:
            cls.__runnable_methods = []
            for _, function in inspect.getmembers(
                cls,
                predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
            ):
                if hasattr(function, RUNNABLE_METHOD_MARK):
                    cls.__runnable_methods.append(
                        getattr(function, RUNNABLE_METHOD_MARK)
                    )
        return cls.__runnable_methods


"""
import bentoml


class SampleRunnable(Runnable):
    SUPPORT_NVIDIA_GPU = True
    SUPPORT_MULTIPLE_CPU_THREADS = True

    def __init__(self) -> None:

        self._model = bentoml.pytorch.load_model(model)

    @bentoml.runnable.methed()
    def predict(self, input1, input2):
        return self._model.predict(input1, input2)

    def __del__(self) -> None:
        del self._model


@Runnable.method(input_spec=bentoml.io_types.NDArray)
def predict():
    pass

"""


@attr.define()
class RunnableMethodConfig(t.Generic[InputType, OutputType]):
    batchable: bool = attr.field()
    batch_dim: int = attr.field()
    input_spec: t.Union[
        LazyType[InputType],
        t.Tuple[LazyType[InputType], ...],
        None,
    ] = attr.field()
    output_spec: t.Optional[LazyType[OutputType]] = attr.field()


RUNNABLE_METHOD_MARK: str = "_bentoml_runnable_method"


def method_decorator(
    meth: WrappedMethod,
    batchable: bool,
    batch_dim: int,
    input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
    output_spec: t.Optional[LazyType[t.Any]] = None,
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
