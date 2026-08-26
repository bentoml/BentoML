from __future__ import annotations

from starlette.testclient import TestClient

import bentoml
from bentoml.exceptions import InvalidArgument


@bentoml.service(metrics={"enabled": False})
class ErroringService:
    @bentoml.api
    def fail(self, x: int) -> int:
        raise InvalidArgument("bad value")


def test_client_error_response_is_json_object_not_array():
    # A 4xx BentoMLException must serialize to a JSON object ``{"error": ...}``,
    # matching the declared OpenAPI error schema (see #4232), not a JSON array.
    # A stray trailing comma previously wrapped the payload in a 1-tuple, which
    # ``JSONResponse`` serialized as ``[{"error": ...}]``.
    with TestClient(app=ErroringService.to_asgi()) as client:
        resp = client.post("/fail", json={"x": 1})

    assert resp.status_code == 400
    body = resp.json()
    assert isinstance(body, dict), (
        f"expected a JSON object, got {type(body).__name__}: {body!r}"
    )
    assert "bad value" in body["error"]
