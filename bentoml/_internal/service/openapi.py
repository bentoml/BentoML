import typing
from collections import OrderedDict

if typing.TYPE_CHECKING:
    from . import Service

HEATHZ_DESC = (
    "Health check endpoint. Expecting an empty response with status code "
    "200 when the service is in health state. The /healthz endpoint is "
    "deprecated (since Kubernetes v1.16)"
)
LIVEZ_DESC = (
    "Health check endpoint for Kubernetes. Healthy endpoint responses with "
    "a 200 OK status."
)
READYZ_DESC = (
    "A 200 OK status from /readyz endpoint indicated the service is ready "
    "to accept traffic. From that point and onward Kubernetes will use "
    "/livez endpoint to perform periodic health checks"
)
METRICS_DESC = "Prometheus metrics endpoint"


def get_service_openapi_doc(svc: "Service"):
    info = OrderedDict(
        title=svc.name,
        description="A Prediction Service built with BentoML",
    )
    if svc.version:
        info["version"] = svc.version
    docs = OrderedDict(
        openapi="3.0.0",
        info=info,
        tags=[{"name": "infra"}, {"name": "app"}],
    )

    paths = OrderedDict()
    default_response = {"200": {"description": "success"}}

    paths["/healthz"] = OrderedDict(
        get=OrderedDict(
            tags=["infra"],
            description=HEATHZ_DESC,
            responses=default_response,
        )
    )
    paths["/livez"] = OrderedDict(
        get=OrderedDict(
            tags=["infra"],
            description=LIVEZ_DESC,
            responses=default_response,
        )
    )
    paths["/readyz"] = OrderedDict(
        get=OrderedDict(
            tags=["infra"],
            description=READYZ_DESC,
            responses=default_response,
        )
    )
    paths["/metrics"] = OrderedDict(
        get=OrderedDict(
            tags=["infra"],
            description=METRICS_DESC,
            responses=default_response,
        )
    )

    for api in svc._apis.values():
        api_path = api.route if api.route.startswith("/") else f"/{api.route}"

        paths[api_path] = {}
        paths[api_path]["post"] = OrderedDict(
            tags=["app"],
            summary=f"Inference API '{api}' under service '{svc.name}'",
            description=api.doc,
            operationId=f"{svc.name}__{api.name}",
            requestBody=dict(content=api.input.openapi_request_schema()),
            responses={
                "200": {
                    "description": "success",
                    "content": api.output.openapi_responses_schema(),
                }
            },
            # examples=None,
            # headers=None,
        )

    docs["paths"] = paths
    return docs
