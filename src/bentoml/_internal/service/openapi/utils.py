from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING
from functools import lru_cache

from bentoml.exceptions import NotFound
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import InternalServerError

from ...utils import LazyLoader
from ...utils.pkg import pkg_version_info
from .specification import Schema

if TYPE_CHECKING:
    import pydantic
    import pydantic.schema as schema

    if pkg_version_info("pydantic")[0] >= 2:
        import pydantic.json_schema as jschema

    from bentoml.exceptions import BentoMLException

else:
    _exc_msg = (
        "Missing required dependency: 'pydantic'. Install with 'pip install pydantic'."
    )
    pydantic = LazyLoader("pydantic", globals(), "pydantic", exc_msg=_exc_msg)
    schema = LazyLoader("schema", globals(), "pydantic.schema", exc_msg=_exc_msg)
    jschema = LazyLoader(
        "jschema", globals(), "pydantic.json_schema", exc_msg="Pydantic v2 is required."
    )

REF_PREFIX = "#/components/schemas/"


def pydantic_components_schema(pydantic_model: type[pydantic.BaseModel]):
    if pkg_version_info("pydantic")[0] >= 2:
        return pydantic_v2_components_schema(pydantic_model)
    else:
        return pydantic_v1_components_schema(pydantic_model)


def pydantic_v2_components_schema(pydantic_model: type[pydantic.BaseModel]):
    json_schema = jschema.model_json_schema(
        pydantic_model, ref_template=REF_PREFIX + "{model}"
    )
    defs = json_schema.pop("$defs", None)
    components: dict[str, Schema] = {pydantic_model.__name__: Schema(**json_schema)}
    if defs is not None:
        # NOTE: This is a nested models, hence we will update the definitions
        components.update({k: Schema(**v) for k, v in defs.items()})
    return components


def pydantic_v1_components_schema(pydantic_model: type[pydantic.BaseModel]):
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
