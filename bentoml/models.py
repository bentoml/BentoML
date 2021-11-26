import sys
import typing as t
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING

import fs
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import Model
from ._internal.types import Tag

if TYPE_CHECKING:  # pragma: no cover
    from ._internal.models import ModelStore, SysPathModel
    from ._internal.runner import Runner

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

_T = t.TypeVar("_T")


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.List["SysPathModel"]:
    return _model_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "SysPathModel":
    return _model_store.get(tag)


@inject
def delete(
    tag: t.Union[Tag, str],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    _model_store.delete(tag)


def import_model(path: str) -> Model:
    return Model.from_fs(fs.open_fs(path))


def export_model(tag: t.Union[Tag, str], path: str):
    model = get(tag)
    model.export(path)


def push(tag: t.Union[Tag, str]):
    model = get(tag)
    model.push()


def pull() -> Model:
    raise NotImplementedError


# fmt: off
class _CreateModelProtocol(t.ContextManager[Model], Protocol):
    def __call__(  # noqa: E704
        self,
        name: str,
        *,
        module: str = ...,
        labels: t.Optional[t.Dict[str, t.Any]] = ...,
        options: t.Optional[t.Dict[str, t.Any]] = ...,
        metadata: t.Optional[t.Dict[str, t.Any]] = ...,
        framework_context: t.Optional[t.Dict[str, t.Any]] = ...,
        _model_store: "ModelStore" = ...,
    ) -> Model: ...
    def __next__(self) -> t.Iterator[Model]: ...  # noqa: E704
    def __enter__(self) -> Model: ...  # noqa: E704
    def __exit__(  # noqa: E704
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> None: ...
# fmt: on


class GeneratorContextManager(t.ContextManager[_T], t.Generic[_T]):
    def __call__(self, func: t.Callable[..., _T]) -> t.Callable[..., _T]:
        ...


@inject
@contextmanager
def create(
    name: str,
    *,
    module: str = "",
    labels: t.Optional[t.Dict[str, t.Any]] = None,
    options: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    framework_context: t.Optional[t.Dict[str, t.Any]] = None,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _CreateModelProtocol:
    res = Model.create(
        name,
        module=module,
        labels=labels,
        options=options,
        metadata=metadata,
        framework_context=framework_context,
    )
    try:
        yield res
    finally:
        res.save(_model_store)


def load_runner(tag: t.Union[Tag, str]) -> "Runner":
    model = get(tag)
    return model.load_runner()


__all__ = [
    "list",
    "get",
    "delete",
    "import_model",
    "export_model",
    "push",
    "pull",
    "load_runner",
]
