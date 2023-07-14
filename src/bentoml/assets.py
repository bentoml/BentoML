from __future__ import annotations

import shutil
import os
import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from .exceptions import BentoMLException, NotFound
from .models import create as create_asset
from ._internal.tag import Tag
from ._internal.models import Model
from ._internal.models import ModelContext
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ._internal.models import ModelStore
    from ._internal.cloud import BentoCloudClient


MODULE_NAME = "bentoml.asset"
ASSET_PREFIX = "bentoml_reserved_assets___"
API_VERSION = "v1"

def convert_tag(tag: t.Union[Tag, str]) -> Tag:
    asset_tag = ASSET_PREFIX + str(tag)
    return Tag.from_taglike(asset_tag)


@inject
def delete(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    tag = convert_tag(tag)
    _model_store.delete(tag)


def import_from_file(
        name: Tag | str,
        path: str | os.PathLike[str],
        *,
        labels: dict[str, str] | None = None,
) ->  Model:

    tag = Tag.from_taglike(name)
    tag = convert_tag(tag)
    context = ModelContext(framework_name="asset", framework_versions={"asset": "1.0"})

    with create_asset(
        tag,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures={},
        labels=labels,
        options=None,
        custom_objects=None,
        context=context,
    ) as bento_asset:
        filename_path = bento_asset.path_of("filename.txt")
        filename = os.path.basename(path)
        dest_path = bento_asset.path_of(filename)
        shutil.copyfile(path, dest_path, follow_symlinks=True)

        with open(filename_path, "w") as f:
            f.write(filename)

        return bento_asset


@inject
def push(
    tag: t.Union[Tag, str],
    *,
    force: bool = False,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
):
    new_tag = convert_tag(tag)
    model_obj = _model_store.get(new_tag)
    if not model_obj:
        raise BentoMLException(f"Asset {tag} not found in local store")
    _cloud_client.push_model(model_obj, force=force)


@inject
def pull(
    tag: t.Union[Tag, str],
    *,
    force: bool = False,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Model:
    tag = convert_tag(tag)
    return _cloud_client.pull_model(tag, force=force)


@inject
def get_file_path(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    tag = convert_tag(tag)
    try:
        bento_model = _model_store.get(tag)
    except NotFound:
        pull(tag)
        bento_model = _model_store.get(tag)
    with open(bento_model.path_of("filename.txt"), "r") as f:
        filename = f.read().strip()

    return bento_model.path_of(filename)


__all__ = [
    "get_file_path",
    "delete",
    "import_from_file",
    "push",
    "pull",
]
