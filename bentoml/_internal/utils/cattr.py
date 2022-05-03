from __future__ import annotations

from datetime import datetime

import cattr
from attr import fields
from cattr import override
from cattr.gen import AttributeOverride

bentoml_cattr = cattr.Converter()

# bentoml_cattr = cattr.Converter(prefer_attrib_converters=True)


def omit_if_init_false(cls) -> dict[str, AttributeOverride]:
    return {f.name: override(omit=True) for f in fields(cls) if not f.init}


def omit_if_default(cls) -> dict[str, AttributeOverride]:
    return {f.name: override(omit_if_default=True) for f in fields(cls)}


def datetime_decoder(dt_like, _):
    if isinstance(dt_like, str):
        return datetime.fromisoformat(dt_like)
    elif isinstance(dt_like, datetime):
        return dt_like
    else:
        raise Exception(f"Unable to parse datetime from '{dt_like}'")


bentoml_cattr.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
bentoml_cattr.register_structure_hook(datetime, datetime_decoder)
