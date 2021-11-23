"""
User facing python APIs for managing local bentos and build new bentos
"""
import logging
import os
import typing as t
from typing import TYPE_CHECKING

import attr
import fs
from simple_di import Provide, inject

from ._internal.bento import Bento, SysPathBento
from ._internal.configuration.containers import BentoMLContainer
from ._internal.types import Tag

if TYPE_CHECKING:  # pragma: no cover
    from ._internal.bento import BentoStore
    from ._internal.models.store import ModelStore


logger = logging.getLogger(__name__)


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> t.List[SysPathBento]:
    return _bento_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    return _bento_store.get(tag)


def delete(
    tag: t.Union[Tag, str],
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    _bento_store.delete(tag)


def import_bento(path: str) -> Bento:
    return Bento.from_fs(fs.open_fs(path))


def export_bento(tag: t.Union[Tag, str], path: str):
    bento = get(tag)
    bento.export(path)


def push(tag: t.Union[Tag, str]):
    bento = get(tag)
    bento.push()


def pull(tag: t.Union[Tag, str]):
    raise NotImplementedError


@inject
def build(
    svc_import_str: str,
    version: t.Optional[str] = None,
    description: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    additional_models: t.Optional[t.List[str]] = None,
    docker: t.Optional[t.Dict[str, t.Any]] = None,
    python: t.Optional[t.Dict[str, t.Any]] = None,
    conda: t.Optional[t.Dict[str, t.Any]] = None,
    build_ctx: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Bento:
    """
    Build a Bento for this Service. A Bento is a file archive containing all the
    specifications, source code, and model files required to run and operate this
    service in production.
    """  # noqa: LN001
    bento = Bento.create(
        svc_import_str,
        build_ctx,
        additional_models,
        version,
        description,
        include,
        exclude,
        docker,
        python,
        conda,
        labels,
        _model_store,
    )
    bento.save(_bento_store)
    logger.info("Bento build success, %s created", bento)
    return bento


def build_from_bentofile(
    bentofile: t.Optional,
    version: t.Optional[str] = None,
    build_ctx: t.Optional[str] = None,
):
    pass


__all__ = [
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "push",
    "pull",
    "build",
]
