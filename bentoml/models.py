import typing as t
from typing import TYPE_CHECKING
from contextlib import contextmanager

import fs
from simple_di import inject, Provide

from ._internal.types import Tag
from ._internal.models import Model
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:  # pragma: no cover
    from ._internal.models import ModelStore, SysPathModel
    from ._internal.runner import Runner


@inject
def list(
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
) -> Model:
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
