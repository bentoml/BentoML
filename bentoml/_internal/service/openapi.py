import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Service
    from ..io_descriptors import IODescriptor


HEALTHZ_DESC = (
    "Health check endpoint. Expecting an empty response with status code"
    " <code>200</code> when the service is in health state. The <code>/healthz</code>"
    " endpoint is <b>deprecated</b> (since Kubernetes v1.16)"
)

LIVEZ_DESC = (
    "Health check endpoint for Kubernetes. Healthy endpoint responses with a"
    " <code>200</code> OK status."
)

READYZ_DESC = (
    "A <code>200</code> OK status from <code>/readyz</code> endpoint indicated"
    " the service is ready to accept traffic. From that point and onward, Kubernetes"
    " will use <code>/livez</code> endpoint to perform periodic health checks."
)

METRICS_DESC = "Prometheus metrics endpoint"


def _generate_responses_schema(
    output: "IODescriptor[t.Any]",
) -> t.Dict[str, t.Dict[str, t.Any]]:
    resp = {
        "200": dict(
            description="success",
            content=output.openapi_responses_schema(),
        ),
        "400": dict(description="Bad Request"),
        "404": dict(description="Not Found"),
        "500": dict(description="Internal Server Error"),
    }
    return resp


def get_service_openapi_doc(svc: "Service"):
    # TODO: add licensing options for service swagger?
    info = {
        "title": svc.name,
        "description": "A Prediction Service built with BentoML",
        "contact": {"email": "contact@bentoml.ai"},
        "version": svc.version or "0.0.0",
    }
    docs: t.Dict[str, t.Any] = {
        "openapi": "3.0.0",
        "info": info,
        "tags": [
            {
                "name": "infra",
                "description": "Infrastructure endpoints",
            },
            {
                "name": "app",
                "description": "Inference endpoints",
            },
        ],
    }

    paths = {}
    default_response = {"200": {"description": "success"}}

    paths["/healthz"] = {
        "get": {
            "tags": ["infra"],
            "description": HEALTHZ_DESC,
            "responses": default_response,
        }
    }
    paths["/livez"] = {
        "get": {
            "tags": ["infra"],
            "description": LIVEZ_DESC,
            "responses": default_response,
        }
    }
    paths["/readyz"] = {
        "get": {
            "tags": ["infra"],
            "description": READYZ_DESC,
            "responses": default_response,
        }
    }
    paths["/metrics"] = {
        "get": {
            "tags": ["infra"],
            "description": METRICS_DESC,
            "responses": default_response,
        }
    }

    for api in svc.apis.values():
        api_path = api.route if api.route.startswith("/") else f"/{api.route}"

        paths[api_path] = {
            "post": dict(
                tags=["app"],
                summary=f"{api}",
                description=api.doc or "",
                operationId=f"{svc.name}__{api.name}",
                requestBody=dict(content=api.input.openapi_request_schema()),
                responses=_generate_responses_schema(api.output),
                # examples=None,
                # headers=None,
            )
        }

    docs["paths"] = paths
    return docs
