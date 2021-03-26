import time
import pytest
from bentoml import config


@pytest.mark.skipif(
    not config("tracing").get("opentracing_server_address"),
    reason="Opentracing is not configured",
)
def test_initialize_tracer():
    import opentracing
    from bentoml.tracing.opentrace import initialize_tracer

    service_name = 'test service name'

    tracer = (
        initialize_tracer(service_name=service_name) or opentracing.global_tracer()
    )  # noqa: F821, E501

    assert tracer is not None


@pytest.mark.skipif(
    not config("tracing").get("opentracing_server_address"),
    reason="Opentracing is not configured",
)
def test_trace():
    from bentoml.tracing.opentrace import trace

    server_url = None
    request_headers = None
    async_transport = False
    sample_rate = 1.0
    standalone = False
    is_root = False
    service_name = "test service"
    span_name = "test service procedure"
    port = 0

    with trace(
        server_url=server_url,
        request_headers=request_headers,
        async_transport=async_transport,
        sample_rate=sample_rate,
        standalone=standalone,
        is_root=is_root,
        service_name=service_name,
        span_name=span_name,
        port=port,
    ) as scope:
        assert scope is not None
        assert scope.span.operation_name == span_name
        assert scope.span.start_time <= time.time()
        assert scope.span.finished is False
