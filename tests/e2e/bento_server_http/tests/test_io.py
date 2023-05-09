from __future__ import annotations

import io
import json
import typing as t

import numpy as np
import pytest
import aiohttp

import bentoml
from bentoml.testing.http import parse_multipart_form

if t.TYPE_CHECKING:
    import PIL.Image as PILImage

    F = t.Callable[..., t.Coroutine[t.Any, t.Any, t.Any]]
else:
    from bentoml._internal.utils import LazyLoader

    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")


@pytest.mark.asyncio
async def test_numpy(arequest: F):
    await arequest(
        api_name="predict_ndarray_enforce_shape",
        headers={"Content-Type": "application/json"},
        data=np.array([[1, 2], [3, 4]]),
        assert_output=np.array([[2, 4], [6, 8]]),
    )
    await arequest(
        api_name="predict_ndarray_enforce_shape",
        headers={"Content-Type": "application/json"},
        data=np.array([1, 2, 3, 4]),
        assert_exception=bentoml.exceptions.BadInput,
        assert_exception_match="NumpyNdarray: Expecting ndarray of shape *",
    )

    await arequest(
        api_name="predict_ndarray_enforce_dtype",
        data=np.array([[2, 1], [4, 3]], dtype=np.uint8),
        assert_output=np.array([[4, 2], [8, 6]], dtype="str"),
    )
    await arequest(
        api_name="predict_ndarray_enforce_dtype",
        headers={"Content-Type": "application/json"},
        data=np.array([["2f", 1], [4, 3]]),
        assert_exception=bentoml.exceptions.BadInput,
        assert_exception_match='Expecting ndarray of dtype "uint8", but "<U21" was received.',
    )


@pytest.mark.asyncio
async def test_json(arequest: F):
    ORIGIN = "http://bentoml.ai"

    await arequest(
        api_name="echo_json",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='"hi"',
        assert_output='"hi"',
    )
    await arequest(
        api_name="echo_json_sync",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='"hi"',
    )

    await arequest(
        api_name="echo_json_enforce_structure",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data={"name": "test", "endpoints": ["predict", "health"]},
        assert_output={"name": "test", "endpoints": ["predict", "health"]},
    )

    from service import ValidateSchema

    # test sending pydantic model
    await arequest(
        api_name="echo_json_enforce_structure",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data=ValidateSchema(name="test", endpoints=["predict", "health"]),
        assert_output={"name": "test", "endpoints": ["predict", "health"]},
    )


@pytest.mark.asyncio
async def test_obj(arequest: F):
    for obj in [1, 2.2, "str", [1, 2, 3], {"a": 1, "b": 2}]:
        obj_str = json.dumps(obj, separators=(",", ":"))
        await arequest(api_name="echo_obj", data=obj_str, assert_output=obj_str)


@pytest.mark.asyncio
async def test_pandas(arequest: F, host: str):
    import pandas as pd

    ORIGIN = "http://bentoml.ai"
    df = pd.DataFrame([[101]], columns=["col1"])
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")

    resp = await client.async_request(
        "POST",
        "/predict_dataframe",
        headers=(("Content-Type", "application/octet-stream"), ("Origin", ORIGIN)),
        data=df.to_parquet(),
    )
    assert resp.ok and resp.status == 200
    assert await resp.read() == b'[{"col1":202}]'

    resp = await client.async_request(
        "POST",
        "/predict_dataframe",
        headers=(("Content-Type", "text/csv"), ("Origin", ORIGIN)),
        data=df.to_csv(),
    )
    assert resp.ok and resp.status == 200
    assert await resp.read() == b'[{"col1":202}]'

    await arequest(
        api_name="predict_dataframe",
        data=df,
        assert_output=lambda out: True
        if all((out == pd.DataFrame([{"col1": 202}])).all())
        else False,
    )


@pytest.mark.asyncio
async def test_file(arequest: F, bin_file: str, host: str):
    # Test File as binary
    with open(str(bin_file), "rb") as f:
        b = f.read()

    await arequest(
        api_name="predict_file",
        data=b,
        headers={"Content-Type": "application/octet-stream"},
        assert_output=b"\x810\x899",
    )

    client = bentoml.client.HTTPClient.from_url(f"http://{host}")

    # Test File as multipart binary
    form = aiohttp.FormData()
    form.add_field("file", b, content_type="application/octet-stream")

    resp = await client.async_request(
        "POST",
        "/predict_file",
        data=form,
    )
    assert resp.ok and resp.status == 200
    assert await resp.read() == b"\x810\x899"

    await arequest(
        api_name="predict_file",
        data=b,
        headers={"Content-Type": "application/pdf"},
        assert_output=b"\x810\x899",
    )


@pytest.mark.asyncio
async def test_image(arequest: F, img_file: str, host: str):
    await arequest(
        api_name="echo_image",
        data=PILImage.open(img_file),
        headers={"Content-Type": "image/bmp"},
    )

    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    with open(str(img_file), "rb") as f1:
        res = await client.async_request(
            "POST",
            "/echo_image",
            data=f1.read(),
            headers={"Content-Type": "application/json"},
        )
        assert res.status == 400

    with open(str(img_file), "rb") as f1:
        b = f1.read()

    bio = io.BytesIO(b)
    bio.name = "test.bmp"

    res = await arequest(
        api_name="echo_image",
        data=PILImage.open(bio),
        headers={"Content-Type": "application/pdf"},
        assert_exception=bentoml.exceptions.BentoMLException,
        assert_exception_match="BentoService error handling API request: mime type application/pdf is not allowed *",
    )


@pytest.fixture(name="img_form_data")
def fixture_img_form_data(img_file: str):
    with open(img_file, "rb") as f1, open(img_file, "rb") as f2:
        form = aiohttp.FormData()
        form.add_field("original", f1.read(), content_type="image/bmp")
        form.add_field("compared", f2.read(), content_type="image/bmp")
    yield form


@pytest.mark.asyncio
async def test_multipart_image_io(host: str, img_form_data: aiohttp.FormData):
    from starlette.datastructures import Headers
    from starlette.datastructures import UploadFile

    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    resp = await client.async_request(
        "POST",
        "/predict_multi_images",
        data=img_form_data,
    )
    assert resp.ok and resp.status == 200

    form = await parse_multipart_form(
        headers=Headers(headers=resp.headers), body=await resp.read()
    )
    for _, v in form.items():
        assert isinstance(v, UploadFile)
        img = PILImage.open(v.file)
        assert np.array(img).shape == (10, 10, 3)


@pytest.mark.asyncio
async def test_multipart_different_args(host: str, img_form_data: aiohttp.FormData):
    client = bentoml.client.HTTPClient.from_url(f"http://{host}")
    resp = await client.async_request(
        "POST",
        "/predict_different_args",
        data=img_form_data,
    )
    assert resp.ok and resp.status == 200
