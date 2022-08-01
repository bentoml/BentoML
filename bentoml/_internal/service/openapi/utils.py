from __future__ import annotations

import typing as t
import inspect
from http import HTTPStatus
from typing import TYPE_CHECKING
from functools import lru_cache

from bentoml.exceptions import BadInput
from bentoml.exceptions import NotFound
from bentoml.exceptions import StateException
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import InternalServerError
from bentoml._internal.types import evaluate_forwardref
from bentoml._internal.service.inference_api import InferenceAPI

from ...utils import LazyLoader
from .specification import Schema
from .specification import Response
from .specification import MediaType
from .specification import Parameter
from .specification import Reference
from .specification import Components

if TYPE_CHECKING:
    import pydantic
    import pydantic.schema as schema

    from bentoml.exceptions import BentoMLException

    from ...service.service import Service
else:
    _exc_msg = (
        "Missing required dependency: 'pydantic'. Install with 'pip install pydantic'."
    )
    pydantic = LazyLoader("pydantic", globals(), "pydantic", exc_msg=_exc_msg)
    schema = LazyLoader("schema", globals(), "pydantic.schema", exc_msg=_exc_msg)

BadRequestType = t.Union[BadInput, InvalidArgument, StateException]
ExceptionType = t.Union[BadRequestType, NotFound, InternalServerError]

REF_PREFIX = "#/components/schemas/"
SUCCESS_DESCRIPTION = "Successful Response"


def generate_model_schema(pydantic_model: t.Type[pydantic.BaseModel]):
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
    return {k: definitions[k] for k in sorted(definitions)}


def generate_components(svc: Service) -> Components:
    return Components(schemas={**exception_components_schema()})


def exception_schema(ex: t.Type[BentoMLException]) -> t.Iterable[FilledExceptionSchema]:
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


@lru_cache(maxsize=1)
def exception_components_schema() -> dict[str, Schema]:
    return {
        schema.title: schema
        for ex in [InvalidArgument, NotFound, InternalServerError]
        for schema in exception_schema(ex)
    }


def generate_responses(api_response: Response) -> dict[int, Response]:
    # This will return a responses following OpenAPI spec.
    # example: {200: {"description": ..., "content": ...}, 404: {"description": ..., "content": ...}, ...}
    return {
        HTTPStatus.OK.value: api_response,
        **{
            ex.error_code.value: Response(
                description=filled.description,
                content={
                    "application/json": MediaType(
                        schema=Reference(f"{REF_PREFIX}{filled.title}")
                    )
                },
            )
            for ex in [InvalidArgument, NotFound, InternalServerError]
            for filled in exception_schema(ex)
        },
    }


def get_typed_annotation(param: inspect.Parameter, globns: dict[str, t.Any]) -> type:
    if isinstance(param.annotation, str):
        return evaluate_forwardref(
            t.ForwardRef(param.annotation), globalns=globns, localns=globns
        )
    return param.annotation


def handle_typed_signatures(fn: t.Callable[..., t.Any]) -> inspect.Signature:
    sig = inspect.signature(fn)
    globns = getattr(fn, "__globals__", {})
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param, globns),
        )
        for param in sig.parameters.values()
    ]
    return inspect.Signature(typed_params)


# To generate Parameter, we are going to do this two ways.
# 1. Use signature of a given function to infer the correct types
# 2. pass to IODescriptor to handle types
def handle_parameters(api: InferenceAPI) -> list[Parameter | Reference]:
    # get users defined signatures
    sig = handle_typed_signatures(api.func)
    for param_name, param in sig.parameters.items():
        print(param_name, param.annotation, param.default)


class FilledExceptionSchema(Schema):
    title: str
    description: str
