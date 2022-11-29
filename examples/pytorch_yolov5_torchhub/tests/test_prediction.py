# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import io
import json

import aiohttp
import numpy as np
import pytest
from bentoml.testing.utils import async_request, parse_multipart_form


@pytest.fixture()
def img_bytes():
    img_path = f"bus.jpg"
    with open(img_path, "rb") as f:
        return f.read()


def check_output(out: bytes) -> bool:
    obj = json.loads(out)
    return any(d["obj"] == "bus" for d in obj[0])


@pytest.mark.asyncio
async def test_image(host, img_bytes):
    await async_request(
        "POST",
        f"http://{host}/predict_image",
        data=img_bytes,
        headers={"Content-Type": "image/png"},
        assert_status=200,
        assert_data=check_output,
    )
