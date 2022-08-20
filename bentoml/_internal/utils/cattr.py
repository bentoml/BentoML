from __future__ import annotations

import typing as t
from datetime import datetime

import attr
from cattr.gen import override
from cattr.gen import make_dict_structure_fn
from cattr.gen import make_dict_unstructure_fn

from .pkg import pkg_version_info

if pkg_version_info("cattrs")[:2] >= (22, 2):
    from cattr import Converter
else:
    from cattr import GenConverter as Converter

bentoml_cattr = Converter(omit_if_default=True)


def datetime_structure_hook(dt_like: str | datetime, _: t.Any) -> datetime:
    if isinstance(dt_like, str):
        return datetime.fromisoformat(dt_like)
    elif isinstance(dt_like, datetime):
        return dt_like
    else:
        raise Exception(f"Unable to parse datetime from '{dt_like}'")


def register_global_structure_hook() -> None:
    from ..tag import Tag
    from ..bento.bento import BentoInfo
    from ..models.model import ModelInfo
    from ..bento.build_config import PythonOptions
    from ..bento.build_config import python_options_structure_hook
    from ..service.openapi.specification import openapi_structure_rename_hook
    from ..service.openapi.specification import openapi_preserve_cls_structure_hook

    bentoml_cattr.register_structure_hook_factory(
        attr.has,
        lambda cls: make_dict_structure_fn(
            cls,
            bentoml_cattr,
            _cattrs_forbid_extra_keys=getattr(cls, "__forbid_extra_keys__", False),
        ),
    )
    bentoml_cattr.register_structure_hook(datetime, datetime_structure_hook)
    bentoml_cattr.register_structure_hook(PythonOptions, python_options_structure_hook)
    bentoml_cattr.register_structure_hook(Tag, lambda d, _: Tag.from_taglike(d))
    bentoml_cattr.register_structure_hook_func(
        lambda cls: issubclass(cls, ModelInfo),
        make_dict_structure_fn(
            ModelInfo,
            bentoml_cattr,
            name=override(omit=True),
            version=override(omit=True),
            _options=override(rename="options"),
        ),
    )
    bentoml_cattr.register_structure_hook_func(
        lambda cls: issubclass(cls, BentoInfo),
        make_dict_structure_fn(
            BentoInfo,
            bentoml_cattr,
            name=override(omit=True),
            version=override(omit=True),
        ),
    )
    # handles all OpenAPI class that includes __rename_fields__
    bentoml_cattr.register_structure_hook_func(
        lambda cls: attr.has(cls) and hasattr(cls, "__rename_fields__"),
        lambda data, cl: openapi_structure_rename_hook(data, cl),
    )
    bentoml_cattr.register_structure_hook_func(
        lambda cls: attr.has(cls) and hasattr(cls, "__preserve_cls_structure__"),
        lambda data, cls: openapi_preserve_cls_structure_hook(data, cls),
    )


def register_global_unstructure_hooks() -> None:
    from ..tag import Tag
    from ..bento.bento import BentoInfo
    from ..models.model import ModelInfo
    from ..models.model import ModelSignature
    from ..models.model import model_signature_unstructure_hook

    bentoml_cattr.register_unstructure_hook_factory(
        attr.has,
        lambda cls: make_dict_unstructure_fn(
            cls,
            bentoml_cattr,
            _cattrs_omit_if_default=getattr(cls, "__omit_if_default__", False),
        ),
    )
    bentoml_cattr.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
    bentoml_cattr.register_unstructure_hook(Tag, str)
    bentoml_cattr.register_unstructure_hook(
        ModelSignature, model_signature_unstructure_hook
    )
    bentoml_cattr.register_unstructure_hook_func(
        lambda cls: issubclass(cls, ModelInfo),
        # Ignore tag, tag is saved via the name and version field
        make_dict_unstructure_fn(
            ModelInfo,
            bentoml_cattr,
            tag=override(omit=True),
            _options=override(rename="options"),
            _cached_module=override(omit=True),
            _cached_options=override(omit=True),
        ),
    )
    bentoml_cattr.register_unstructure_hook(
        BentoInfo,
        # Ignore tag, tag is saved via the name and version field
        make_dict_unstructure_fn(BentoInfo, bentoml_cattr, tag=override(omit=True)),
    )
    bentoml_cattr.register_unstructure_hook_factory(
        lambda cls: attr.has(cls) and hasattr(cls, "__rename_fields__"),
        lambda cls: make_dict_unstructure_fn(
            cls,
            bentoml_cattr,
            # for all classes under OpenAPI, we want to omit default values.
            _cattrs_omit_if_default=getattr(cls, "__omit_if_default__", True),
            **{k: override(rename=v) for k, v in cls.__rename_fields__.items()},
        ),
    )
