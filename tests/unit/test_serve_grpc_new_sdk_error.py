"""Regression test for bentoml#5607.

`serve-grpc` used to fail with the unhelpful message::

    <class '_bentoml_sdk.service.factory.Service'> type doesn't support gRPC serving

when pointed at a service defined with the `@bentoml.service` decorator
(the bentoml>=1.2 programming model). gRPC is only implemented for the
legacy `bentoml.Service` API, so this case should raise a clear error
that tells the user what to do instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bentoml.exceptions import BentoMLException
from bentoml.serving import serve_grpc_production

NEW_SDK_SERVICE = """
from __future__ import annotations

import bentoml


@bentoml.service(name="grpc_smoke")
class Supervisor:
    @bentoml.api
    def greet(self, name: str = "world") -> str:
        return f"hello {name}"
"""


def test_serve_grpc_rejects_new_sdk_service_with_clear_message(
    tmp_path: Path,
) -> None:
    (tmp_path / "service.py").write_text(NEW_SDK_SERVICE)

    with pytest.raises(BentoMLException) as exc_info:
        serve_grpc_production(
            "service.py:Supervisor",
            working_dir=str(tmp_path),
        )

    message = str(exc_info.value)
    assert "gRPC serving is not supported" in message
    assert "@bentoml.service" in message
    assert "bentoml serve" in message
