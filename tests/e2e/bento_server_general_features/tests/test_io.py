# type: ignore[no-untyped-def]

import io
import sys
import json

import numpy as np
import pytest
import aiohttp

from bentoml.testing.utils import async_request
from bentoml.testing.utils import parse_multipart_form


@pytest.mark.asyncio
async def test_numpy(host):
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_shape",
        headers={"Content-Type": "application/json"},
        data="[[1,2],[3,4]]",
        assert_status=200,
        assert_data=b"[[2, 4], [6, 8]]",
    )
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_shape",
        headers={"Content-Type": "application/json"},
        data="[1,2,3,4]",
        assert_status=400,
    )
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_dtype",
        headers={"Content-Type": "application/json"},
        data="[[2,1],[4,3]]",
        assert_status=200,
        assert_data=b'[["4", "2"], ["8", "6"]]',
    )
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_dtype",
        headers={"Content-Type": "application/json"},
        data='[["2f",1],[4,3]]',
        assert_status=400,
    )


@pytest.mark.asyncio
async def test_json(host):
    ORIGIN = "http://bentoml.ai"

    await async_request(
        "POST",
        f"http://{host}/echo_json",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='"hi"',
        assert_status=200,
        assert_data=b'"hi"',
    )

    await async_request(
        "POST",
        f"http://{host}/echo_json_sync",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='"hi"',
        assert_status=200,
        assert_data=b'"hi"',
    )

    await async_request(
        "POST",
        f"http://{host}/echo_json_enforce_structure",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='{"name":"test","endpoints":["predict","health"]}',
        assert_status=200,
        assert_data=b'{"name":"test","endpoints":["predict","health"]}',
    )


@pytest.mark.asyncio
async def test_obj(host):
    for obj in [1, 2.2, "str", [1, 2, 3], {"a": 1, "b": 2}]:
        obj_str = json.dumps(obj, separators=(",", ":"))
        await async_request(
            "POST",
            f"http://{host}/echo_obj",
            headers=(("Content-Type", "application/json"),),
            data=obj_str,
            assert_status=200,
            assert_data=obj_str.encode("utf-8"),
        )


@pytest.mark.asyncio
async def test_pandas(host):
    import pandas as pd

    ORIGIN = "http://bentoml.ai"

    df = pd.DataFrame([[101]], columns=["col1"])

    await async_request(
        "POST",
        f"http://{host}/predict_dataframe",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data=df.to_json(orient="records"),
        assert_status=200,
        assert_data=b'[{"col1":202}]',
    )

    # pyarrow only support python 3.7+
    if sys.version_info >= (3, 7):
        await async_request(
            "POST",
            f"http://{host}/predict_dataframe",
            headers=(("Content-Type", "application/octet-stream"), ("Origin", ORIGIN)),
            data=df.to_parquet(),
            assert_status=200,
            assert_data=b'[{"col1":202}]',
        )

    await async_request(
        "POST",
        f"http://{host}/predict_dataframe",
        headers=(("Content-Type", "text/csv"), ("Origin", ORIGIN)),
        data=df.to_csv(),
        assert_status=200,
        assert_data=b'[{"col1":202}]',
    )


@pytest.mark.asyncio
async def test_file(host, bin_file):
    # Test File as binary
    with open(str(bin_file), "rb") as f:
        b = f.read()

    await async_request(
        "POST",
        f"http://{host}/predict_file",
        data=b,
        headers={"Content-Type": "application/octet-stream"},
        assert_data=b"\x810\x899",
    )

    # Test File as multipart binary
    form = aiohttp.FormData()
    form.add_field("file", b, content_type="application/octet-stream")

    await async_request(
        "POST",
        f"http://{host}/predict_file",
        data=form,
        assert_data=b"\x810\x899",
    )

    # Test Exception
    await async_request(
        "POST",
        f"http://{host}/predict_file",
        data=b,
        headers={"Content-Type": "application/pdf"},
        assert_status=500,
    )


@pytest.mark.asyncio
async def test_image(host, img_file):
    import PIL.Image

    with open(str(img_file), "rb") as f1:
        img_bytes = f1.read()

    status, headers, body = await async_request(
        "POST",
        f"http://{host}/echo_image",
        data=img_bytes,
        headers={"Content-Type": "image/bmp"},
    )
    assert status == 200
    assert headers["Content-Type"] == "image/bmp"

    bio = io.BytesIO(body)
    bio.name = "test.bmp"
    img = PIL.Image.open(bio)
    array1 = np.array(img)
    array2 = PIL.Image.open(img_file)

    np.testing.assert_array_almost_equal(array1, array2)

    await async_request(
        "POST",
        f"http://{host}/echo_image",
        data=img_bytes,
        headers={"Content-Type": "application/json"},
        assert_status=400,
    )

    # Test Exception
    with open(str(img_file), "rb") as f1:
        b = f1.read()
    await async_request(
        "POST",
        f"http://{host}/echo_image",
        data=b,
        headers={"Content-Type": "application/pdf"},
        assert_status=400,
    )


# SklearnRunner is not suppose to take multiple arguments
# TODO: move e2e tests to use a new bentoml.PickleModel module
@pytest.mark.skip
@pytest.mark.asyncio
async def test_multipart_image_io(host, img_file):
    import PIL.Image
    from starlette.datastructures import UploadFile

    with open(img_file, "rb") as f1:
        with open(img_file, "rb") as f2:
            form = aiohttp.FormData()
            form.add_field("original", f1.read(), content_type="image/bmp")
            form.add_field("compared", f2.read(), content_type="image/bmp")

    status, headers, body = await async_request(
        "POST",
        f"http://{host}/predict_multi_images",
        data=form,
    )

    assert status == 200

    form = await parse_multipart_form(headers=headers, body=body)
    for _, v in form.items():
        assert isinstance(v, UploadFile)
        img = PIL.Image.open(v.file)
        assert np.array(img).shape == (10, 10, 3)
