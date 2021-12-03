import sys
import typing as t
from types import TracebackType
from typing import TYPE_CHECKING
from contextlib import contextmanager

import fs
from simple_di import inject
from simple_di import Provide

from ._internal.types import Tag
from ._internal.models import Model
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ._internal.models import ModelStore
    from ._internal.runner import Runner

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

_T = t.TypeVar("_T")


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.List["Model"]:
    return _model_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Model":
    return _model_store.get(tag)


@inject
def delete(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    _model_store.delete(tag)


@inject
def import_model(
    path: str, *, _model_store: "ModelStore" = Provide[BentoMLContainer.model_store]
) -> Model:
    return Model.from_fs(fs.open_fs(path)).save(_model_store)


def export_model(
    tag: t.Union[Tag, str],
    path: str,
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    model = get(tag, _model_store=_model_store)
    model.export(path)


def push(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    raise NotImplementedError


def pull(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Model:
    raise NotImplementedError


# fmt: off
class _CreateModelProtocol(Protocol):
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
    def __exit__(  # noqa: E704,E301
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_val: t.Optional[BaseException],
        exc_tb: t.Optional[TracebackType],
    ) -> None: ...
# fmt: on


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


@inject
def load_runner(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Runner":
    model = get(tag, _model_store=_model_store)
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
