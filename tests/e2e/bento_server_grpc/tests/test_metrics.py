from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from bentoml.client import Client
from bentoml.grpc.utils import import_generated_stubs

if TYPE_CHECKING:
    from bentoml.grpc.v1 import service_pb2 as pb
else:
    pb, _ = import_generated_stubs()


@pytest.mark.asyncio
async def test_metrics_available(host: str):
    client = Client.from_url(host)
    resp = await client.async_predict_multi_images(
        original=np.random.randint(255, size=(10, 10, 3)).astype("uint8"),
        compared=np.random.randint(255, size=(10, 10, 3)).astype("uint8"),
    )
    assert isinstance(resp, pb.Response)
    resp = await client.async_ensure_metrics_are_registered("input_data")
    assert isinstance(resp, pb.Response)
