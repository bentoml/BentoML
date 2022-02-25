# pylint: disable=redefined-outer-name
# type: ignore[no-untyped-def]

import io

import numpy as np
import pytest
import aiohttp

from bentoml.io import PandasDataFrame
from bentoml.testing.utils import async_request
from bentoml.testing.utils import parse_multipart_form


@pytest.fixture()
def img_file(tmpdir):
    import PIL.Image

    img_file_ = tmpdir.join("test_img.bmp")
    img = PIL.Image.fromarray(np.random.randint(255, size=(10, 10, 3)).astype("uint8"))
    img.save(str(img_file_))
    return str(img_file_)


@pytest.fixture()
def bin_file(tmpdir):
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)


@pytest.mark.asyncio
async def test_numpy(host):
    await async_request(
        "POST",
        f"http://{host}/predict_ndarray_enforce_shape",
        headers={"Content-Type": "application/json"},
        data="[[1,2],[3,4]]",
        assert_status=200,
        assert_data=b"[[2, 4, 6, 8]]",
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
        data='["hi"]',
        assert_status=200,
        assert_data=b'["hi"]',
    )

    await async_request(
        "POST",
        f"http://{host}/pydantic_json",
        headers=(("Content-Type", "application/json"), ("Origin", ORIGIN)),
        data='{"name":"test","endpoints":["predict","health"]}',
        assert_status=200,
        assert_data=b'{"name":"test","endpoints":["predict","health"]}',
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
        assert_data=b"[202]",
    )

    if PandasDataFrame.parquet_engine:
        await async_request(
            "POST",
            f"http://{host}/predict_dataframe",
            headers=(("Content-Type", "application/octet-stream"), ("Origin", ORIGIN)),
            data=df.to_parquet(engine=PandasDataFrame.parquet_engine),
            assert_status=200,
            assert_data=b"[202]",
        )

    await async_request(
        "POST",
        f"http://{host}/predict_dataframe",
        headers=(("Content-Type", "text/csv"), ("Origin", ORIGIN)),
        data=df.to_csv(),
        assert_status=200,
        assert_data=b"[202]",
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
