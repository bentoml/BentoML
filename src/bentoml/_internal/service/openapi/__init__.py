from __future__ import annotations

import typing as t
from functools import lru_cache
from http import HTTPStatus
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Union

import fastapi
from fastapi.openapi.utils import get_openapi

from bentoml.exceptions import InternalServerError
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import NotFound

from ...types import LazyType
from ...utils.cattr import bentoml_cattr
from ...utils.merge import deep_merge as _deep_merge
from .specification import Components
from .specification import Contact
from .specification import Info
from .specification import MediaType
from .specification import OpenAPISpecification
from .specification import Operation
from .specification import PathItem
from .specification import Reference
from .specification import Response
from .specification import Tag
from .utils import REF_PREFIX
from .utils import exception_components_schema
from .utils import exception_schema

if TYPE_CHECKING:
    from .. import Service
    from ..inference_api import InferenceAPI
    from .specification import Schema

# Type hints for external functions
OpenAPIRoutes = List[Any]  # FastAPI route type is complex, use Any for now
GetOpenAPIFunc = Callable[[str, str, List[Any]], Dict[str, Any]]
_get_openapi: GetOpenAPIFunc = t.cast(GetOpenAPIFunc, get_openapi)

# Type hints for cattrs converter
T = TypeVar("T")
StructureFunc = Callable[[Any, type[T]], T]
bentoml_cattr: Any  # type: ignore # Complex cattrs type, ignore for now

# Type hints for deep_merge
MergeDict = Dict[str, Any]
DeepMergeFunc = Callable[[MergeDict, MergeDict], MergeDict]
_deep_merge: DeepMergeFunc = t.cast(DeepMergeFunc, _deep_merge)

OpenAPIDict = dict[str, t.Any]
OpenAPIPaths = dict[str, PathItem]
OpenAPIPathsDict = dict[str, dict[str, t.Any]]

SUCCESS_DESCRIPTION = "Successful Response"

INFRA_DECRIPTION = {
    "/healthz": "Health check endpoint. Expecting an empty response with status code <code>200</code> when the service is in health state. The <code>/healthz</code> endpoint is <b>deprecated</b>. (since Kubernetes v1.16)",
    "/livez": "Health check endpoint for Kubernetes. Healthy endpoint responses with a <code>200</code> OK status.",
    "/readyz": "A <code>200</code> OK status from <code>/readyz</code> endpoint indicated the service is ready to accept traffic. From that point and onward, Kubernetes will use <code>/livez</code> endpoint to perform periodic health checks.",
    "/metrics": "Prometheus metrics endpoint. The <code>/metrics</code> responses with a <code>200</code>. The output can then be used by a Prometheus sidecar to scrape the metrics of the service.",
}

__all__ = ["generate_spec"]

INFRA_TAG = Tag(
    name="Infrastructure",
    description="Common infrastructure endpoints for observability.",
)
APP_TAG = Tag(
    name="Service APIs", description="BentoML Service API endpoints for inference."
)


def make_api_path(api: InferenceAPI[t.Any]) -> str:
    return api.route if api.route.startswith("/") else f"/{api.route}"


@lru_cache(maxsize=1)
def make_infra_endpoints() -> dict[str, PathItem]:
    return {
        endpoint: PathItem(
            get=Operation(
                responses={"200": Response(description=SUCCESS_DESCRIPTION)},
                tags=[INFRA_TAG.name],
                description=INFRA_DECRIPTION[endpoint],
            )
        )
        for endpoint in INFRA_DECRIPTION
    }


def generate_service_components(svc: Service) -> Components:
    components = Components(schemas={})
    for api in svc.apis.values():
        api_components = Components(schemas={})
        input_components = api.input.openapi_components()
        if input_components and isinstance(input_components, dict):
            merged_schemas = t.cast(
                Dict[str, Dict[str, Union[Schema, Reference]]],
                _deep_merge({"schemas": api_components.schemas}, input_components),
            ).get("schemas", {})
            api_components = Components(schemas=merged_schemas)

        output_components = api.output.openapi_components()
        if output_components and isinstance(output_components, dict):
            merged_schemas = t.cast(
                Dict[str, Dict[str, Union[Schema, Reference]]],
                _deep_merge({"schemas": api_components.schemas}, output_components),
            ).get("schemas", {})
            api_components = Components(schemas=merged_schemas)

        merged_schemas = t.cast(
            Dict[str, Dict[str, Union[Schema, Reference]]],
            _deep_merge(
                {"schemas": components.schemas}, {"schemas": api_components.schemas}
            ),
        ).get("schemas", {})
        components = Components(schemas=merged_schemas)

    # merge exception at last
    merged_schemas = t.cast(
        Dict[str, Dict[str, Union[Schema, Reference]]],
        _deep_merge(
            {"schemas": components.schemas}, {"schemas": exception_components_schema()}
        ),
    ).get("schemas", {})
    return Components(schemas=merged_schemas)


def generate_spec(
    svc: Service, *, openapi_version: str = "3.0.2"
) -> OpenAPISpecification:
    """Generate a OpenAPI specification for a service."""
    mounted_app_paths: OpenAPIPaths = {}
    schema_components = Components(schemas={})
    openapi: OpenAPIDict = {}
    paths: OpenAPIPathsDict = {}
    app: t.Any

    for app, _, _ in svc.mount_apps:
        if LazyType["fastapi.FastAPI"]("fastapi.FastAPI").isinstance(app):
            app = t.cast("fastapi.FastAPI", app)
            routes: OpenAPIRoutes = list(app.routes)
            openapi_dict: OpenAPIDict = _get_openapi(
                str(app.title), str(app.version), routes
            )

            paths = t.cast(OpenAPIPathsDict, openapi_dict["paths"])
            path_items: Dict[str, PathItem] = {
                str(k): bentoml_cattr.structure(v, PathItem) for k, v in paths.items()
            }
            mounted_app_paths.update(path_items)

            if "components" in openapi:
                merged_schemas = t.cast(
                    Dict[str, Dict[str, Union[Schema, Reference]]],
                    _deep_merge(
                        {"schemas": schema_components.schemas},
                        {"schemas": openapi["components"].get("schemas", {})},
                    ),
                ).get("schemas", {})
                schema_components = Components(schemas=merged_schemas)

    service_components = generate_service_components(svc)
    merged_schemas = t.cast(
        Dict[str, Dict[str, Union[Schema, Reference]]],
        _deep_merge(
            {"schemas": schema_components.schemas},
            {"schemas": service_components.schemas},
        ),
    ).get("schemas", {})
    schema_components = Components(schemas=merged_schemas)
    components = (
        Components(schemas=schema_components.schemas)
        if schema_components.schemas
        else None
    )

    return OpenAPISpecification(
        openapi=openapi_version,
        tags=[APP_TAG, INFRA_TAG],
        components=components,
        info=Info(
            title=svc.name,
            description=svc.doc,
            version=svc.tag.version if svc.tag and svc.tag.version else "None",
            contact=Contact(name="BentoML Team", email="contact@bentoml.com"),
        ),
        servers=[{"url": "."}],
        paths={
            # setup infra endpoints
            **make_infra_endpoints(),
            # setup inference endpoints
            **{
                make_api_path(api): PathItem(
                    post={
                        "responses": {
                            HTTPStatus.OK.value: api.output.openapi_responses(),
                            **{
                                ex.error_code.value: Response(
                                    description=filled.description,
                                    content={
                                        "application/json": MediaType(
                                            schema=Reference(
                                                f"{REF_PREFIX}{filled.title}"
                                            )
                                        )
                                    },
                                )
                                for ex in [
                                    InvalidArgument,
                                    NotFound,
                                    InternalServerError,
                                ]
                                for filled in exception_schema(ex)
                            },
                        },
                        "tags": [APP_TAG.name],
                        "consumes": [api.input.mime_type],
                        "produces": [api.output.mime_type],
                        "x-bentoml-name": api.name,
                        "summary": str(api),
                        "description": api.doc or "",
                        "requestBody": api.input.openapi_request_body(),
                        "operationId": f"{svc.name}__{api.name}",
                    },
                )
                for api in svc.apis.values()
            },
            **mounted_app_paths,
        },
    )
