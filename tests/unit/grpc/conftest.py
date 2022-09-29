from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import pytest

if TYPE_CHECKING:
    import grpc
else:
    from bentoml.grpc.utils import import_grpc

    grpc, _ = import_grpc()


@pytest.fixture(scope="module", name="mock_unary_unary_handler")
def fixture_mock_handler() -> MagicMock:
    handler = MagicMock(spec=grpc.RpcMethodHandler)
    handler.request_streaming = PropertyMock(return_value=False)
    handler.response_streaming = PropertyMock(return_value=False)
    return handler
