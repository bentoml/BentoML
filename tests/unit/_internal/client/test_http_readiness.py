from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import Optional
from typing import Union

import httpx
import pytest

import bentoml
from _bentoml_impl.client import http as client_http
from bentoml.exceptions import ServiceUnavailable

Mode = Literal["sync", "async"]
Outcome = Union[int, Literal["timeout", "connect"]]
TimeoutSettings = dict[str, Optional[float]]


@dataclass
class HTTPState:
    now: float = 0.0
    steps: list[tuple[float, Outcome]] = field(
        default_factory=list[tuple[float, Outcome]]
    )
    requests: list[tuple[str, TimeoutSettings]] = field(
        default_factory=list[tuple[str, TimeoutSettings]]
    )

    def monotonic(self) -> float:
        return self.now


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> Mode:
    if request.param == "sync":
        return "sync"
    assert request.param == "async"
    return "async"


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> HTTPState:
    state = HTTPState()
    monkeypatch.setattr(client_http, "time", state)

    def handler(request: httpx.Request) -> httpx.Response:
        state.requests.append((request.url.path, request.extensions["timeout"]))
        if request.url.path == "/schema.json":
            return httpx.Response(200, json={"routes": []})
        assert request.url.path == "/readyz"
        elapsed, outcome = state.steps.pop(0)
        state.now += elapsed
        if outcome == "timeout":
            raise httpx.ReadTimeout("not ready yet", request=request)
        if outcome == "connect":
            raise httpx.ConnectError("not connected yet", request=request)
        return httpx.Response(outcome)

    # The fake clock advances only at the HTTP boundary. MockTransport does not
    # enforce HTTPX timeouts; inspect the request budget and late results here.
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    transport = httpx.MockTransport(handler)

    def transport_for_url(
        _client: httpx.Client | httpx.AsyncClient, _url: httpx.URL
    ) -> httpx.MockTransport:
        return transport

    for client_class in (httpx.Client, httpx.AsyncClient):
        monkeypatch.setattr(client_class, "_transport_for_url", transport_for_url)
    return state


def run_client(
    mode: Mode,
    *,
    direct_timeout: float | None = None,
    timeout: float = 30,
    server_ready_timeout: float | None = None,
) -> None:
    def check_timeout(client: bentoml.SyncHTTPClient | bentoml.AsyncHTTPClient) -> None:
        # Per-probe overrides must not change ordinary endpoint request limits.
        assert client.client.timeout == httpx.Timeout(timeout)

    if mode == "sync":
        with bentoml.SyncHTTPClient(
            "http://bento.test",
            token="",
            timeout=timeout,
            server_ready_timeout=server_ready_timeout,
        ) as client:
            check_timeout(client)
            if direct_timeout is not None:
                client.wait_until_server_ready(direct_timeout)
    else:

        async def run() -> None:
            client = bentoml.AsyncHTTPClient(
                "http://bento.test",
                token="",
                timeout=timeout,
                server_ready_timeout=server_ready_timeout,
            )
            try:
                async with client:
                    check_timeout(client)
                    if direct_timeout is not None:
                        await client.wait_until_server_ready(direct_timeout)
            finally:
                await client.close()

        asyncio.run(run())


def ready_timeouts(state: HTTPState) -> list[float | None]:
    values: list[float | None] = []
    for path, timeout in state.requests:
        if path == "/readyz":
            assert len(set(timeout.values())) == 1
            values.append(timeout["read"])
    return values


def test_readiness_caps_probe_timeout(mode: Mode, mock_http: HTTPState) -> None:
    mock_http.steps = [(0, 200)]
    run_client(mode, server_ready_timeout=5)
    assert ready_timeouts(mock_http) == [5]
    assert mock_http.requests[-1] == ("/schema.json", httpx.Timeout(30).as_dict())


def test_readiness_preserves_shorter_request_timeout(
    mode: Mode, mock_http: HTTPState
) -> None:
    mock_http.steps = [(0, 200)]
    run_client(mode, server_ready_timeout=5, timeout=1)
    assert ready_timeouts(mock_http) == [1]


def test_readiness_retries_use_remaining_budget(
    mode: Mode, mock_http: HTTPState
) -> None:
    mock_http.steps = [(0.5, "timeout"), (1.5, "connect"), (1, 503), (0.1, 200)]
    run_client(mode, server_ready_timeout=5)
    assert ready_timeouts(mock_http) == [5, 4.5, 3, 2]
    assert mock_http.requests[-1][0] == "/schema.json"


def test_readiness_retries_preserve_shorter_request_timeout(
    mode: Mode, mock_http: HTTPState
) -> None:
    mock_http.steps = [(1, "timeout"), (0.5, 503), (0.1, 200)]
    run_client(mode, server_ready_timeout=2, timeout=1)
    assert ready_timeouts(mock_http) == [1, 1, 0.5]


@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("elapsed", [5, 6])
def test_readiness_does_not_accept_late_response(
    mode: Mode, mock_http: HTTPState, status: int, elapsed: float
) -> None:
    mock_http.steps = [(elapsed, status)]
    with pytest.raises(ServiceUnavailable, match="after 5 seconds"):
        run_client(mode, server_ready_timeout=5)
    assert [path for path, _ in mock_http.requests] == ["/readyz"]


def test_readiness_uses_request_timeout_by_default(
    mode: Mode, mock_http: HTTPState
) -> None:
    mock_http.steps = [(0, 200)]
    run_client(mode, timeout=3, server_ready_timeout=None)
    assert ready_timeouts(mock_http) == [3]


@pytest.mark.parametrize("timeout", [0, -1])
def test_nonpositive_readiness_timeout_skips_initial_wait(
    mode: Mode, mock_http: HTTPState, timeout: float
) -> None:
    run_client(mode, server_ready_timeout=timeout)
    assert [path for path, _ in mock_http.requests] == ["/schema.json"]


@pytest.mark.parametrize("timeout", [0, -1])
def test_direct_nonpositive_readiness_timeout_fails_without_request(
    mode: Mode, mock_http: HTTPState, timeout: float
) -> None:
    with pytest.raises(ServiceUnavailable, match=f"after {timeout} seconds"):
        run_client(mode, server_ready_timeout=0, direct_timeout=timeout)
    assert [path for path, _ in mock_http.requests] == ["/schema.json"]
