from __future__ import annotations

from typing import TYPE_CHECKING
from functools import partial
from functools import lru_cache

from .utils import handle_parameters
from .utils import generate_responses
from .utils import generate_service_components
from ...utils import bentoml_cattr
from .specification import Tag
from .specification import Info
from .specification import Contact
from .specification import PathItem
from .specification import Response
from .specification import Apache2_0
from .specification import Operation
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

INFRA_TAG = Tag(name="infra", description="Infrastructure endpoints.")
APP_TAG = Tag(name="app", description="Inference endpoints.")


# setup bentoml default tags
# add a field for users to add additional tags if needed.
def generate_tags(
    *, additional_tags: list[dict[str, str] | Tag] | None = None
) -> list[Tag]:
    defined_tags = [INFRA_TAG, APP_TAG]
    if additional_tags:
        partial_structure = partial(bentoml_cattr.structure, cl=Tag)
        defined_tags.extend(map(partial_structure, additional_tags))

    return defined_tags


def make_api_path(api: InferenceAPI) -> str:
    return api.route if api.route.startswith("/") else f"/{api.route}"


def generate_info(svc: Service, *, term_of_service: str | None = None) -> Info:
    # default version if svc.tag is None
    version = "0.0.0"
    if svc.tag and svc.tag.version:
        version = svc.tag.version

    # TODO: add support for readme via 'description'
    # summary="A BentoService built for inference."
    # description=svc.doc
    return Info(
        title=svc.name,
        description="A BentoService built for inference.",
        version=version,
        termsOfService=term_of_service,
        contact=Contact(name="BentoML Team", email="contact@bentoml.ai"),
        license=Apache2_0,
    )


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


def generate_spec(
    svc: Service,
    *,
    openapi_version: str = "3.0.2",
    additional_tags: list[dict[str, str] | Tag] | None = None,
):
    """Generate a OpenAPI specification for a service."""
    return OpenAPISpecification(
        openapi=openapi_version,
        info=generate_info(svc),
        components=generate_service_components(svc),
        tags=generate_tags(additional_tags=additional_tags),
        paths={
            # setup infra endpoints
            **make_infra_endpoints(),
            # setup inference endpoints
            **{
                make_api_path(api): PathItem(
                    post=Operation(
                        responses=generate_responses(api.output),
                        tags=[APP_TAG.name],
                        summary=f"{api}",
                        description=api.doc or "",
                        parameters=handle_parameters(api),
                        requestBody=api.input.openapi_request_body(),
                        operationId=f"{svc.name}__{api.name}",
                    )
                )
                for api in svc.apis.values()
            },
        },
    )
