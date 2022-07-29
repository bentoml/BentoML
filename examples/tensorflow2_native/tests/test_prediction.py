# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import json
import asyncio

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
        img_arr = np.array(PIL.Image.open(f"samples/{digit}.png"))
        img_arr = img_arr / 255.0
        img_dic = {
            "bytes": img_bytes,
            "array": img_arr,
        }
        images[digit] = img_dic

    return images


@pytest.mark.asyncio
async def test_numpy(host, img_data):
    datas = [json.dumps(d["array"].tolist()) for d in img_data.values()]

    # request one by one
    for data in datas[:-3]:
        await async_request(
            "POST",
            f"http://{host}/predict_ndarray",
            headers={"Content-Type": "application/json"},
            data=datas[0],
            assert_status=200,
        )

    # request all at once, should trigger micro-batch prediction
    tasks = tuple(
        async_request(
            "POST",
            f"http://{host}/predict_ndarray",
            headers={"Content-Type": "application/json"},
            data=data,
            assert_status=200,
        )
        for data in datas[-3:]
    )
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_image(host, img_data):
    for d in img_data.values():
        img_bytes = d["bytes"]
        await async_request(
            "POST",
            f"http://{host}/predict_image",
            data=img_bytes,
            headers={"Content-Type": "image/png"},
            assert_status=200,
        )
