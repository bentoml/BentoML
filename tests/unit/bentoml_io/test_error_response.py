from __future__ import annotations

from http import HTTPStatus

import pytest
from starlette.testclient import TestClient

import bentoml
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import InternalServerError
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import NotFound


class Unauthorized(BentoMLException):
    """No shipped exception maps to 401, so the auth branch needs its own."""

    error_code = HTTPStatus.UNAUTHORIZED


class Forbidden(BentoMLException):
    error_code = HTTPStatus.FORBIDDEN


@bentoml.service(metrics={"enabled": False})
class ErroringService:
    @bentoml.api
    def invalid_argument(self, x: int) -> int:
        raise InvalidArgument("bad value")

    @bentoml.api
    def not_found(self, x: int) -> int:
        raise NotFound("no such thing")

    @bentoml.api
    def unauthorized(self, x: int) -> int:
        raise Unauthorized("nope")

    @bentoml.api
    def forbidden(self, x: int) -> int:
        raise Forbidden("nope")

    @bentoml.api
    def server_error(self, x: int) -> int:
        raise InternalServerError("boom")

    @bentoml.api
    def unhandled(self, x: int) -> int:
        raise ValueError("not a BentoMLException")

    @bentoml.api
    def echo(self, x: int) -> int:
        return x


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app=ErroringService.to_asgi()) as c:
        yield c


def _error_body(client: TestClient, route: str, payload: dict, expected_status: int):
    """POST and assert the error body is a JSON *object* with an ``error`` key.

    ``api_endpoint_wrapper`` builds every error payload as a dict and hands it
    to ``JSONResponse``. A stray trailing comma turns one into a 1-tuple, which
    serialises as a JSON *array* and breaks any client that reads
    ``body["error"]`` — which is what #4232 was.
    """
    resp = client.post(route, json=payload)

    assert resp.status_code == expected_status
    body = resp.json()
    assert isinstance(body, dict), (
        f"{route}: expected a JSON object, got {type(body).__name__}: {body!r}"
    )
    assert "error" in body, f"{route}: no 'error' key in {body!r}"
    assert isinstance(body["error"], str), (
        f"{route}: 'error' should be a string, got {type(body['error']).__name__}"
    )
    return body


class TestClientErrorResponses:
    """#4232 — the branch that was actually broken."""

    def test_400_is_a_json_object_not_an_array(self, client):
        body = _error_body(client, "/invalid_argument", {"x": 1}, 400)

        assert "bad value" in body["error"]

    def test_404_is_a_json_object_not_an_array(self, client):
        # Same `else` branch as 400, via a different status, so the assertion
        # pins the branch rather than one exception type.
        body = _error_body(client, "/not_found", {"x": 1}, 404)

        assert "no such thing" in body["error"]


class TestOtherErrorBranchesAreAlsoObjects:
    """Controls for the branches `api_endpoint_wrapper` handles separately.

    Each builds its own dict literal, so each can independently regress in the
    same way. Without these, a tuple-comma in any of them passes the suite.
    """

    @pytest.mark.parametrize(
        "route,status", [("/unauthorized", 401), ("/forbidden", 403)]
    )
    def test_auth_errors_are_json_objects(self, client, route, status):
        body = _error_body(client, route, {"x": 1}, status)

        # The auth branch deliberately does not leak the exception message.
        assert body["error"] == "Authorization error"
        assert "nope" not in body["error"]

    def test_server_error_is_a_json_object(self, client):
        body = _error_body(client, "/server_error", {"x": 1}, 500)

        # The >=500 branch also withholds detail; the message goes to the log.
        assert "unexpected error" in body["error"].lower()
        assert "boom" not in body["error"]

    def test_unhandled_exception_is_a_json_object(self, client):
        # Not a BentoMLException at all — the bare `except Exception` branch.
        body = _error_body(client, "/unhandled", {"x": 1}, 500)

        assert "unexpected error" in body["error"].lower()
        assert "not a BentoMLException" not in body["error"]

    def test_validation_error_is_a_json_object_with_detail(self, client):
        # Raised by pydantic before the endpoint runs, so it takes the
        # ValidationError branch, which carries an extra `detail` list.
        resp = client.post("/echo", json={"x": "not-an-int"})

        assert resp.status_code == 400
        body = resp.json()
        assert isinstance(body, dict), f"expected a JSON object, got {body!r}"
        assert "validation error" in body["error"]
        assert isinstance(body["detail"], list)


def test_successful_response_is_unaffected(client):
    """Control: the fix must not change the happy path."""
    resp = client.post("/echo", json={"x": 7})

    assert resp.status_code == 200
    assert resp.json() == 7
