from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import APP_TAG
from .utils import INFRA_TAG
from .utils import generate_info
from .utils import generate_tags
from .utils import generate_responses
from .utils import SUCCESS_DESCRIPTION
from .specification import PathItem
from .specification import Response
from .specification import Operation
from .specification import RequestBody
from .specification import OpenAPISpecification

if TYPE_CHECKING:
    from .. import Service

INFRA_DECRIPTION = {
    "/healthz": "Health check endpoint. Expecting an empty response with status code <code>200</code> when the service is in health state. The <code>/healthz</code> endpoint is <b>deprecated</b>. (since Kubernetes v1.16)",
    "/livez": "Health check endpoint for Kubernetes. Healthy endpoint responses with a <code>200</code> OK status.",
    "/readyz": "A <code>200</code> OK status from <code>/readyz</code> endpoint indicated the service is ready to accept traffic. From that point and onward, Kubernetes will use <code>/livez</code> endpoint to perform periodic health checks.",
    "/metrics": "Prometheus metrics endpoint. The <code>/metrics</code> responses with a <code>200</code>. The output can then be used by a Prometheus sidecar to scrape the metrics of the service.",
}


def infra_paths():
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


def generate_spec(svc: Service, *, openapi_version: str = "3.0.2"):
    paths = infra_paths()

    for api in svc.apis.values():
        api_path = api.route if api.route.startswith("/") else f"/{api.route}"
        paths.update(
            {
                api_path: PathItem(
                    post=Operation(
                        responses=generate_responses(api.output),
                        tags=[APP_TAG.name],
                        summary=f"{api}",
                        description=api.doc or "",
                        requestBody=RequestBody(
                            content=api.input.openapi_request_schema()
                        ),
                        operationId=f"{svc.name}__{api.name}",
                    )
                )
            }
        )
    return OpenAPISpecification(
        openapi=openapi_version,
        info=generate_info(svc),
        paths=paths,
        tags=generate_tags(),
    )
