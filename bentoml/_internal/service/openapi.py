import typing as t
from collections import OrderedDict

import attr

if t.TYPE_CHECKING:
    from . import Service
    from .inference_api import InferenceAPI


HEALTHZ_DESC = (
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


@attr.s
class Responses:
    description = attr.ib()
    content = attr.ib()


def _mapval(func: t.Callable[..., t.Any], dict_: t.Dict[str, t.Any], factory=dict):
    seq = factory()
    seq.update(zip(dict_.keys(), map(func, dict_.values())))
    return seq


def _responses(api: "InferenceAPI") -> t.Dict[str, Responses]:
    resp = {
        "200": Responses(
            description="success", content=api.output.openapi_responses_schema()
        ),
        "400": Responses(description="error", content="Bad Request"),
        "500": Responses(description="error", content="Internal Server Error"),
    }
    return _mapval(lambda x: attr.asdict(x), resp)


def get_service_openapi_doc(svc: "Service"):
    info = OrderedDict(
        title=svc.name,
        description="A Prediction Service built with BentoML",
    )
    info["version"] = svc.version if svc.version else "0.0.0"
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
            description=HEALTHZ_DESC,
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
            summary=f"Inference API endpoints '{repr(api)}' under service '{svc.name}'",
            description=api.doc or "",
            operationId=f"{svc.name}__{api.name}",
            requestBody=dict(content=api.input.openapi_request_schema()),
            responses=_responses(api),
            # examples=None,
            # headers=None,
        )

    docs["paths"] = paths
    return docs
