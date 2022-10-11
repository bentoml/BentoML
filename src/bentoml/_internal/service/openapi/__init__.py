from __future__ import annotations

import typing as t
from http import HTTPStatus
from typing import TYPE_CHECKING
from functools import lru_cache

from deepmerge.merger import Merger

from bentoml.exceptions import NotFound
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import InternalServerError

from .utils import REF_PREFIX
from .utils import exception_schema
from .utils import exception_components_schema
from .specification import Tag
from .specification import Info
from .specification import Contact
from .specification import PathItem
from .specification import Response
from .specification import MediaType
from .specification import Operation
from .specification import Reference
from .specification import Components
from .specification import OpenAPISpecification

if TYPE_CHECKING:
    from .. import Service
    from ..inference_api import InferenceAPI

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

merger = Merger(
    # merge dicts
    [(dict, "merge")],
    # override all other types
    ["override"],
    # override conflicting types
    ["override"],
)


def make_api_path(api: InferenceAPI) -> str:
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
    components: dict[str, t.Any] = {}
    for api in svc.apis.values():
        api_components = {}
        input_components = api.input.openapi_components()
        if input_components:
            merger.merge(api_components, input_components)
        output_components = api.output.openapi_components()
        if output_components:
            merger.merge(api_components, output_components)

        merger.merge(components, api_components)

    # merge exception at last
    merger.merge(components, {"schemas": exception_components_schema()})

    return Components(**components)


def generate_spec(svc: Service, *, openapi_version: str = "3.0.2"):
    """Generate a OpenAPI specification for a service."""
    return OpenAPISpecification(
        openapi=openapi_version,
        tags=[APP_TAG, INFRA_TAG],
        components=generate_service_components(svc),
        info=Info(
            title=svc.name,
            summary="A ML Service created with BentoML",
            description=svc.doc,
            version=svc.tag.version if svc.tag and svc.tag.version else "None",
            contact=Contact(name="BentoML Team", email="contact@bentoml.ai"),
        ),
        paths={
            # setup infra endpoints
            **make_infra_endpoints(),
            # setup inference endpoints
            **{
                make_api_path(api): PathItem(
                    post=Operation(
                        responses={
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
                        tags=[APP_TAG.name],
                        summary=f"{api}",
                        description=api.doc or "",
                        requestBody=api.input.openapi_request_body(),
                        operationId=f"{svc.name}__{api.name}",
                    )
                )
                for api in svc.apis.values()
            },
        },
    )
