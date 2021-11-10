# pylint: disable=redefined-outer-name
import dataclasses
import json
import typing as t

import aiohttp
import imageio
import numpy as np
import pydantic
import pytest


@dataclasses.dataclass
class _ExampleSchema:
    name: str
    endpoints: t.List[str]


class _Schema(pydantic.BaseModel):
    name: str
    endpoints: t.List[str]


@pytest.mark.parametrize(
    "obj",
    [
        _ExampleSchema(name="test", endpoints=["predict", "health"]),
        _Schema(name="test", endpoints=["predict", "health"]),
        np.array([[1]]),
    ],
)
def test_json_encoder(obj):
    from bentoml._internal.io_descriptors.json import DefaultJsonEncoder

    dumped = json.dumps(
        obj,
        cls=DefaultJsonEncoder,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    )
    assert (
        dumped == '{"name":"test","endpoints":["predict","health"]}'
        or dumped == "[[1]]"
    )


@pytest.mark.asyncio
async def test_json(host, async_request):
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
async def test_pandas(host, async_request):
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


@pytest.fixture()
def bin_file(tmpdir):
    bin_file_ = tmpdir.join("bin_file.bin")
    with open(bin_file_, "wb") as of:
        of.write("â".encode("gb18030"))
    return str(bin_file_)


@pytest.mark.asyncio
async def test_file(host, bin_file, async_request):
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

    await async_request(
        "POST",
        f"http://{host}/predict_invalid_filetype",
        data=b,
        headers={"Content-Type": "application/octet-stream"},
        assert_status=400,
    )


@pytest.fixture()
def img_file(tmpdir):
    img_file_ = tmpdir.join("test_img.jpg")
    imageio.imwrite(str(img_file_), np.random.randint(2, size=(10, 10, 3)))  # noqa
    return str(img_file_)


@pytest.mark.asyncio
async def test_image(host, img_file, async_request):
    import imageio  # noqa # pylint: disable=unused-import
    import numpy as np  # noqa # pylint: disable=unused-import

    def _verify_image(img_bytes):
        return imageio.imread(img_bytes).shape == (10, 10, 3)

    with open(str(img_file), "rb") as f1:
        with open(str(img_file), "rb") as f2:
            form = aiohttp.FormData()
            form.add_field("original", f1.read(), content_type="image/jpeg")
            form.add_field("compared", f2.read(), content_type="image/jpeg")

            await async_request(
                "POST",
                f"http://{host}/predict_multi_images",
                data=form,
                assert_data=_verify_image,
            )

    with open(str(img_file), "rb") as f1:
        form = aiohttp.FormData()
        form.add_field("original", f1.read(), content_type="image/jpeg")
        await async_request(
            "POST",
            f"http://{host}/echo_image",
            data=form,
            assert_status=200,
        )
        await async_request(
            "POST",
            f"http://{host}/predict_invalid_imgtype",
            data=form,
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
        assert_status=500,
    )


@pytest.mark.asyncio
async def test_multipart(host, img_file, async_request):
    import numpy as np  # noqa # pylint: disable=unused-import

    def _verify_multipart_response(rfc):
        return b"\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF" in rfc

    with open(str(img_file), "rb") as f1:
        with open(str(img_file), "rb") as f2:
            form = aiohttp.FormData()
            form.add_field("original", f1.read(), content_type="image/jpeg")
            form.add_field("compared", f2.read(), content_type="image/jpeg")

            await async_request(
                "POST",
                f"http://{host}/echo_return_multipart",
                data=form,
                assert_data=_verify_multipart_response,
            )
