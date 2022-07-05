from __future__ import annotations

import typing as t
from datetime import datetime

import attr
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


bentoml_cattr.register_structure_hook_factory(
    attr.has,
    lambda cls: make_dict_structure_fn(
        cls,
        bentoml_cattr,
        _cattrs_forbid_extra_keys=getattr(cls, "__forbid_extra_keys__", False),
    ),
)

bentoml_cattr.register_unstructure_hook_factory(
    attr.has,
    lambda cls: make_dict_unstructure_fn(
        cls,
        bentoml_cattr,
        _cattrs_omit_if_default=getattr(cls, "__omit_if_default__", False),
    ),
)

bentoml_cattr.register_structure_hook(datetime, datetime_structure_hook)
bentoml_cattr.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
