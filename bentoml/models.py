import typing as t
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


@inject
@contextmanager
def create(
    name: str,
    *,
    module: str = "",
    labels: t.Optional[t.Dict[str, t.Any]] = None,
    options: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    context: t.Optional[t.Dict[str, t.Any]] = None,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Iterator[Model]:
    res = Model.create(
        name,
        module=module,
        labels=labels,
        options=options,
        metadata=metadata,
        context=context,
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
