from __future__ import annotations

import typing as t
from http import HTTPStatus

from deepmerge.merger import Merger

from bentoml._internal.service.openapi import APP_TAG
from bentoml._internal.service.openapi import INFRA_TAG
from bentoml._internal.service.openapi import make_infra_endpoints
from bentoml._internal.service.openapi.specification import Contact
from bentoml._internal.service.openapi.specification import Info
from bentoml._internal.service.openapi.specification import MediaType
from bentoml._internal.service.openapi.specification import OpenAPISpecification
from bentoml._internal.service.openapi.specification import PathItem
from bentoml._internal.service.openapi.specification import Reference
from bentoml._internal.service.openapi.specification import Response
from bentoml._internal.service.openapi.specification import Schema
from bentoml._internal.service.openapi.utils import exception_components_schema
from bentoml._internal.service.openapi.utils import exception_schema
from bentoml._internal.types import LazyType
from bentoml._internal.utils import bentoml_cattr
from bentoml.exceptions import InternalServerError
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import NotFound

if t.TYPE_CHECKING:
    from .factory import Service

merger = Merger(
    # merge dicts
    [(dict, "merge")],
    # override all other types
    ["override"],
    # override conflicting types
    ["override"],
)

REF_TEMPLATE = "#/components/schemas/{model}"


def generate_spec(svc: Service[t.Any], *, openapi_version: str = "3.0.2"):
    """Generate a OpenAPI specification for a service."""
    mounted_app_paths = {}
    schema_components: dict[str, dict[str, Schema]] = {}

    for app, _, _ in svc.mount_apps:
        if LazyType["fastapi.FastAPI"]("fastapi.FastAPI").isinstance(app):
            from fastapi.openapi.utils import get_openapi

            openapi = get_openapi(
                title=app.title,
                version=app.version,
                routes=app.routes,
            )

            mounted_app_paths.update(
                {
                    k: bentoml_cattr.structure(v, PathItem)
                    for k, v in openapi["paths"].items()
                }
            )

            if "components" in openapi:
                merger.merge(schema_components, openapi["components"])

    merger.merge(schema_components, generate_service_components(svc))

    return OpenAPISpecification(
        openapi=openapi_version,
        tags=[APP_TAG, INFRA_TAG],
        components=schema_components,
        info=Info(
            title=svc.name,
            description=svc.doc,
            version=svc.bento.tag.version or "None" if svc.bento else "None",
            contact=Contact(name="BentoML Team", email="contact@bentoml.com"),
        ),
        servers=[{"url": "."}],
        paths={
            # setup infra endpoints
            **make_infra_endpoints(),
            # setup inference endpoints
            **{
                api.route: PathItem(
                    post={
                        "responses": {
                            HTTPStatus.OK.value: api.openapi_response(),
                            **{
                                ex.error_code.value: Response(
                                    description=field.description,
                                    content={
                                        "application/json": MediaType(
                                            schema=Reference(
                                                REF_TEMPLATE.format(model=field.title)
                                            )
                                        )
                                    },
                                )
                                for ex in [
                                    InvalidArgument,
                                    NotFound,
                                    InternalServerError,
                                ]
                                for field in exception_schema(ex)
                            },
                        },
                        "tags": [APP_TAG.name],
                        "consumes": [api.input_spec.mime_type()],
                        "produces": [api.output_spec.mime_type()],
                        "x-bentoml-name": api.name,
                        "description": api.doc or "",
                        "requestBody": api.openapi_request(),
                        "operationId": f"{svc.name}__{api.name}",
                    },
                )
                for api in svc.apis.values()
            },
            **mounted_app_paths,
        },
    )


def generate_service_components(svc: Service[t.Any]) -> dict[str, t.Any]:
    components: dict[str, t.Any] = {}
    for name, api in svc.apis.items():
        input_components = api.input_spec.openapi_components(name)
        components.update(input_components)
        output_components = api.output_spec.openapi_components(name)
        components.update(output_components)

    # merge exception at last
    components.update(exception_components_schema())
    return {"schemas": components}
