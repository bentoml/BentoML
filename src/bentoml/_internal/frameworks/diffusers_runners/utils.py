from __future__ import annotations

import logging

import diffusers
from huggingface_hub import model_info

import bentoml
from bentoml import Tag
from bentoml._internal.frameworks.diffusers import _str2cls
from bentoml._internal.frameworks.diffusers import import_model
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import NotFound

logger = logging.getLogger(__name__)


def construct_bentoml_model_name(model_name: str, model_id: str, backend: str = "pt"):
    name = "-".join([backend, model_name, model_id])
    name = name.replace("/", "--").replace("_", "-")
    return name


def get_model_or_download(
    model_name: str,
    model_id: str,
    use_available: bool = False,
    pipeline_class: type[diffusers.DiffusionPipeline] = diffusers.DiffusionPipeline,
    pipeline_mapping: dict[str, type[diffusers.DiffusionPipeline]] | None = None,
) -> bentoml.Model:
    bentoml_model_name = construct_bentoml_model_name(model_name, model_id)

    available_tag: Tag | None = None

    try:
        model = bentoml.diffusers.get(bentoml_model_name)
        available_tag = model.tag
        if use_available:
            return model

    except NotFound:
        pass

    # get newest model commit hash
    info = model_info(model_id)
    newest_sha = info.sha
    if available_tag and newest_sha == available_tag.version:
        return bentoml.diffusers.get(bentoml_model_name)

    logger.info(f"{bentoml_model_name} not in model store, try to import")

    if isinstance(pipeline_class, str):
        pipeline_class = resolve_pipeline_class(pipeline_class, pipeline_mapping)

    model = import_model(
        bentoml_model_name,
        model_id,
        pipeline_class=pipeline_class,
        sync_with_hub_version=True,
    )

    return model


def resolve_pipeline_class(
    pipeline_str: str,
    pipeline_mapping: dict[str, type[diffusers.DiffusionPipeline]] | None = None,
) -> type[diffusers.DiffusionPipeline]:
    try:
        pipeline_class = _str2cls(pipeline_str)
    except AttributeError:
        try:
            if pipeline_mapping:
                pipeline_class = pipeline_mapping[pipeline_str.lower()]
            else:
                raise KeyError
        except KeyError:
            raise BentoMLException(f"Cannot resolve pipeline {pipeline_str}")

    return pipeline_class
