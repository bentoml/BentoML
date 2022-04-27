import abc
import typing as t
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    InputType = t.TypeVar("InputType")
    OutputType = t.TypeVar("OutputType")

    WrappedMethod = t.TypeVar("WrappedMethod", bound=t.Callable[..., t.Any])

import attr

from bentoml._internal.types import LazyType

logger = logging.getLogger(__name__)


class Runnable(abc.ABC):
    ALLOW_DEVICES: t.List[str]
    ALLOW_MULTIPLE_CPU_THREADS: bool

    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @abc.abstractmethod
    def __del__(self) -> None:
        ...

    @staticmethod
    def method(
        batchable: bool,
        batch_dim: int,
        input_spec: t.Union[
            LazyType[InputType],
            t.Tuple[LazyType[InputType]],
            None,
        ] = None,
        output_spec: t.Optional[LazyType[OutputType]] = None,
    ) -> RunnableMethod[InputType, OutputType]:
        pass


import bentoml


class SampleRunnable(Runnable):
    ALLOW_DEVICES = ["cpu", "nvidia.com/gpu"]
    ALLOW_MULTIPLE_CPU_THREADS = True

    def __init__(self) -> None:

        self._model = bentoml.pytorch.load_model(model)

    @bentoml.runnable.methed()
    def predict(self, input1, input2):
        return self._model.predict(input1, input2)

    def __del__(self) -> None:
        del self._model


@attr.define()
class RunnableMethodCinfigs(t.Generic[InputType, OutputType]):
    batchable: bool = attr.field()
    batch_dim: int = attr.field()
    input_spec: t.Union[
        LazyType[InputType],
        t.Tuple[LazyType[InputType]],
        None,
    ] = attr.field()
    output_spec: t.Optional[LazyType[OutputType]] = attr.field()


def method_decorator(
    meth: WrappedMethod,
    batchable: bool,
    batch_dim: int,
    input_spec: t.Union[
        LazyType[InputType],
        t.Tuple[LazyType[InputType]],
        None,
    ] = None,
    output_spec: t.Optional[LazyType[OutputType]] = None,
) -> RunnableMethod[InputType, OutputType]:
    return RunnableMethod(
        func.__name__,
        batchable=batchable,
        batch_dim=batch_dim,
        input_spec=input_spec,
        output_spec=output_spec,
    )
    return meth


def get_runnable_methods(
    runnable_class: t.Type[Runnable],
) -> t.List[RunnableMethod[t.Any, t.Any]]:
    return [
        v for v in runnable_class.__dict__.values() if isinstance(v, RunnableMethod)
    ]
