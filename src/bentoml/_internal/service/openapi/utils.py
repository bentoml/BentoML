from __future__ import annotations

import sys
import typing as t
from typing import get_args
from typing import TypedDict
from typing import get_origin
from typing import TYPE_CHECKING
from functools import lru_cache

if sys.version_info >= (3, 10):

    from types import UnionType

    def is_union(tp):  # check Union[A,B] or A | B
        return tp == t.Union or get_origin(tp) == UnionType

else:

    def is_union(tp):  # check Union[A,B]
        return tp == t.Union


if sys.version_info >= (3, 11):
    from typing import Required
    from typing import NotRequired
else:
    from typing_extensions import Required
    from typing_extensions import NotRequired

if sys.version_info >= (3, 10):
    from typing import is_typeddict
    from typing import get_type_hints
else:
    from typing_extensions import is_typeddict
    from typing_extensions import get_type_hints

from bentoml.exceptions import NotFound
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import InternalServerError

from ...utils import LazyLoader
from .specification import Schema

if TYPE_CHECKING:
    import pydantic
    import pydantic.schema as schema

    from bentoml.exceptions import BentoMLException

else:
    _exc_msg = (
        "Missing required dependency: 'pydantic'. Install with 'pip install pydantic'."
    )
    pydantic = LazyLoader("pydantic", globals(), "pydantic", exc_msg=_exc_msg)
    schema = LazyLoader("schema", globals(), "pydantic.schema", exc_msg=_exc_msg)

REF_PREFIX = "#/components/schemas/"


def pydantic_components_schema(pydantic_model: t.Type[pydantic.BaseModel]):
    flat_models = schema.get_flat_models_from_model(pydantic_model)
    model_name_map = schema.get_model_name_map(flat_models)

    # gets model definitions
    definitions: dict[str, dict[str, t.Any]] = {}
    for model in flat_models:
        m_schema, m_definitions, _ = schema.model_process_schema(
            model, model_name_map=model_name_map, ref_prefix=REF_PREFIX
        )
        definitions.update(m_definitions)
        model_name = model_name_map[model]
        definitions[model_name] = m_schema
    return {k: Schema(**definitions[k]) for k in sorted(definitions)}


def typeddict_components_schema(typeddict: t.Type[TypedDict]):
    typeddict_set = get_flat_typeddicts_from_typeddict(
        typeddict=typeddict, typeddict_set=set()
    )
    definitions: dict[str, Schema] = {}
    for typeddict in typeddict_set:
        definitions[typeddict.__name__] = typed_dict_to_dict(typeddict)
    return {k: Schema(**definitions[k]) for k in sorted(definitions)}


def get_flat_typeddicts_from_typeddict(
    typeddict: t.Type[TypedDict], typeddict_set: t.Set[t.Type[t.Any]]
) -> t.Set[t.Type[t.Any]]:
    typeddict_set.add(typeddict)
    field_types: t.Dict[str, t.Any] = get_type_hints(typeddict, include_extras=True)

    for _, field_type in field_types.items():
        if is_typeddict(field_type) is True:
            typeddict_set.union(
                get_flat_typeddicts_from_typeddict(
                    typeddict=field_type, typeddict_set=typeddict_set
                )
            )

    return typeddict_set


@lru_cache(maxsize=1)
def exception_components_schema() -> dict[str, Schema]:
    return {
        schema.title: schema
        for ex in [InvalidArgument, NotFound, InternalServerError]
        for schema in exception_schema(ex)
    }


def exception_schema(ex: t.Type[BentoMLException]) -> t.Iterable[FilledExceptionSchema]:
    # convert BentoML exception to OpenAPI components schema
    error_properties = {
        "msg": Schema(title="Message", type="string"),
        "type": Schema(title="Error Type", type="string"),
    }

    yield FilledExceptionSchema(
        title=ex.__name__,
        type="object",
        description=ex.error_code.phrase,
        properties=error_properties,
        required=list(error_properties),
    )


class FilledExceptionSchema(Schema):
    title: str
    description: str


def python_type_to_openapi_type(
    python_type: t.Any, field_name: str
) -> t.Union[str, dict[str, t.Any]]:
    _python_type = getattr(python_type, "__origin__", python_type)
    if _python_type == int:
        return {
            "title": field_name,
            "type": "integer",
        }
    elif _python_type == float:
        return {
            "title": field_name,
            "type": "number",
        }
    elif _python_type == bool:
        return {
            "title": field_name,
            "type": "boolean",
        }
    elif _python_type == str:
        return {
            "title": field_name,
            "type": "string",
        }
    elif is_typeddict(python_type) is True:
        return {"$ref": f"{REF_PREFIX}{python_type.__name__}"}

    elif _python_type == list or _python_type is set:
        items_type = python_type_to_openapi_type(python_type.__args__[0], "")["type"]
        return {"title": field_name, "type": "array", "items": {"type": items_type}}
    elif _python_type == tuple:
        items_types = [
            {"type": python_type_to_openapi_type(_item_type, "")["type"]}
            for _item_type in python_type.__args__
        ]
        return {"title": field_name, "type": "array", "items": items_types}
    elif _python_type == dict:
        return {
            "title": field_name,
            "type": "object",
            "additionalProperties": {
                "type": python_type_to_openapi_type(python_type.__args__[1], "")["type"]
            },
        }
    elif is_union(_python_type):
        types = [
            python_type_to_openapi_type(x, "")["type"] for x in python_type.__args__
        ]
        return {"oneOf": [{"type": t} for t in types]}
    elif _python_type == t.Any:
        return {"title": field_name, "type": "any"}
    elif _python_type == type(None):
        return {"title": field_name, "type": "null"}
    else:
        return {}


def typed_dict_to_dict(typeddict: t.Type[TypedDict]) -> t.Dict[str, t.Any]:
    dict_types: t.Dict[str, t.Any] = get_type_hints(typeddict, include_extras=True)
    required_fields: t.List[str] = [
        field_name
        for field_name, v in dict_types.items()
        if get_origin(v) is not NotRequired
    ]

    fields_type: t.Dict[str, t.Any] = {}
    for field_name, v in dict_types.items():
        if (get_origin(v) is Required or get_origin(v) is NotRequired) and get_args(v):
            fields_type[field_name] = python_type_to_openapi_type(
                get_args(v)[0], field_name
            )
        else:
            fields_type[field_name] = python_type_to_openapi_type(v, field_name)

    return {
        "type": "object",
        "title": typeddict.__name__,
        "required": required_fields,
        "properties": {
            field_name: {
                "title": field_name,
                "type": field_type,
            }
            if isinstance(fields_type, str)
            else field_type
            for field_name, field_type in fields_type.items()
        },
    }
