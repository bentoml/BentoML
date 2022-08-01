from __future__ import annotations

import typing as t
from http import HTTPStatus
from typing import TYPE_CHECKING

from bentoml.exceptions import BadInput
from bentoml.exceptions import NotFound
from bentoml.exceptions import StateException
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import InternalServerError

from ...utils import LazyLoader
from .specification import Tag
from .specification import Info
from .specification import Schema
from .specification import Contact
from .specification import Response
from .specification import Apache2_0
from .specification import MediaType
from .specification import Reference

if TYPE_CHECKING:
    import pydantic

    from bentoml.exceptions import BentoMLException

    from ...io_descriptors import IODescriptor
    from ...service.service import Service
else:
    pydantic = LazyLoader(
        "pydantic",
        globals(),
        "pydantic",
        exc_msg="Missing required dependency: 'pydantic'. Install with 'pip install pydantic'.",
    )

BadRequestType = t.Union[BadInput, InvalidArgument, StateException]
ExceptionType = t.Union[BadRequestType, NotFound, InternalServerError]

REF_PREFIX = "#/components/schemas/"
SUCCESS_DESCRIPTION = "Successful Response"


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


def generate_exception_components_schema() -> dict[str, Schema]:
    return {
        schema.title: schema
        for ex in [InvalidArgument, NotFound, InternalServerError]
        for schema in exception_schema(ex)
    }


def generate_responses(output: IODescriptor[t.Any]) -> dict[int, Response]:
    # This will return a responses following OpenAPI spec.
    # example: {200: {"description": ..., "content": ...}, 404: {"description": ..., "content": ...}, ...}
    return {
        HTTPStatus.OK.value: Response(
            description=SUCCESS_DESCRIPTION,
            content=output.openapi_responses_schema(),
        ),
        **{
            ex.error_code.value: Response(
                description=filled_schema.description,
                content={
                    "application/json": MediaType(
                        schema=Reference(f"{REF_PREFIX}{filled_schema.title}")
                    )
                },
            )
            for ex in [InvalidArgument, NotFound, InternalServerError]
            for filled_schema in exception_schema(ex)
        },
    }


# setup bentoml default tags
# add a field for users to add additional tags if needed.
INFRA_TAG = Tag(name="infra", description=f"Infrastructure endpoints.")
APP_TAG = Tag(name="app", description="Inference endpoints.")


def generate_tags(
    *, additional_tags: list[dict[str, str] | Tag] | None = None
) -> list[Tag]:
    defined_tags = [INFRA_TAG, APP_TAG]
    if additional_tags:
        defined_tags.extend(map(Tag.from_taglike, additional_tags))
    return defined_tags


def generate_info(svc: Service, *, term_of_service: str | None = None) -> Info:
    # default version if svc.tag is None
    version = "0.0.0"
    if svc.tag and svc.tag.version:
        version = svc.tag.version

    return Info(
        title=svc.name,
        description="A BentoService built for inference.",
        # summary = "A BentoService built for inference.",
        # description=svc.doc, # TODO: support display readmes.
        version=version,
        termsOfService=term_of_service,
        contact=Contact(name="BentoML Team", email="contact@bentoml.ai"),
        license=Apache2_0,
    )


def generate_components():
    ...


class FilledExceptionSchema(Schema):
    title: str
    description: str
