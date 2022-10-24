from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest

import bentoml
from bentoml.testing.grpc import create_channel
from bentoml.testing.grpc import async_client_call

if TYPE_CHECKING:
    import jax.numpy as jnp


@pytest.fixture()
def img():
    import PIL.Image

    images = {}
    digits = list(range(10))
    for digit in digits:
        img_path = f"samples/{digit}.png"
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        arr = np.array(PIL.Image.open(io.BytesIO(img_bytes)))
        images[digit] = {
            "bytes": img_bytes,
            "array": arr,
        }

    return images


@pytest.fixture(name="client")
@pytest.mark.asyncio
async def fixture_client(host: str):
    return await bentoml.client.from_url(host)


# TODO: update with bentoml.client once
# gRPC client is implemented.


@pytest.mark.asyncio
async def test_image_grpc(
    host: str, img: dict[int, dict[str, bytes | jnp.ndarray]], enable_grpc: bool
):
    if not enable_grpc:
        pytest.skip("Skipping gRPC test when testing on HTTP.")
    async with create_channel(host) as channel:
        for digit, d in img.items():
            img_bytes = d["bytes"]
            await async_client_call(
                "predict",
                channel=channel,
                data={"serialized_bytes": img_bytes},
                assert_data=lambda resp: resp.ndarray.int32_values == [digit],
            )


@pytest.mark.asyncio
async def test_image_http(
    client: bentoml.client.Client,
    img: dict[int, dict[str, bytes | jnp.ndarray]],
    enable_grpc: bool,
):
    if enable_grpc:
        pytest.skip("Skipping HTTP test when testing on gRPC.")
    for digit, d in img.items():
        img_bytes = d["bytes"]
        expected = f"{digit}".encode()
        assert await client.predict(img_bytes) == expected
