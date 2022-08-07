# pylint: disable=redefined-outer-name

import io
import json

import numpy as np
import pytest

from bentoml.testing.utils import async_request


@pytest.fixture()
def img_data():
    import PIL.Image

    images = {}
    digits = list(range(10))
    for digit in digits:
        img_path = f"samples/{digit}.png"
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_arr = np.array(PIL.Image.open(io.BytesIO(img_bytes)))
        img_arr = img_arr / 255.0
        img_dic = {
            "bytes": img_bytes,
            "array": img_arr,
        }
        images[digit] = img_dic

    return images


@pytest.mark.asyncio
async def test_numpy(host, img_data):
    for digit, d in img_data.items():
        img_arr = d["array"]
        img_arr_json = json.dumps(img_arr.tolist())
        bdigit = f"{digit}".encode()
        await async_request(
            "POST",
            f"http://{host}/predict_ndarray",
            headers={"Content-Type": "application/json"},
            data=img_arr_json,
            assert_status=200,
            assert_data=bdigit,
        )


@pytest.mark.asyncio
async def test_image(host, img_data):
    for digit, d in img_data.items():
        img_bytes = d["bytes"]
        bdigit = f"{digit}".encode()
        await async_request(
            "POST",
            f"http://{host}/predict_image",
            data=img_bytes,
            headers={"Content-Type": "image/png"},
            assert_status=200,
            assert_data=bdigit,
        )
