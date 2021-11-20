import typing as t
from typing import TYPE_CHECKING

import fs
import fs.mirror
from simple_di import Provide, inject

import bentoml

from ._internal.configuration.containers import BentoMLContainer
from ._internal.model import Model
from ._internal.types import Tag

if TYPE_CHECKING:  # pragma: no cover
    from ._internal.model import ModelStore


@inject
def list(
    tag: t.Optional[t.Union[Tag, str]] = None,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.List[Model]:
    return _model_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Model:
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
    fs.mirror.mirror(model.fs, fs.open_fs(path), copy_if_newer=False)
    pass


def push(tag: t.Union[Tag, str]):
    model = get(tag)
    model.push()


def pull() -> Model:
    pass


def load_runner(tag: t.Union[Tag, str]) -> bentoml.Runner:
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
