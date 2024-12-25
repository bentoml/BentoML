from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING
from typing import Dict
from typing import Tuple

import numpy as np
import pytest

from bentoml.client import AsyncHTTPClient
from bentoml.testing.utils import parse_multipart_form

if TYPE_CHECKING:
    import PIL.Image as PILImage
else:
    from bentoml._internal.utils.lazy_loader import LazyLoader

    PILImage = LazyLoader("PILImage", globals(), "PIL.Image")


@pytest.mark.asyncio
async def test_numpy(host: str):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        data = json.dumps([[1, 2], [3, 4]])
        response = await client.client.post(
            "/predict_ndarray_enforce_shape",
            headers={"Content-Type": "application/json"},
            data=data,
        )
        assert response.status_code == 200
        assert await response.aread() == b"[[2, 4], [6, 8]]"

        data = json.dumps([[1, 2], [3, 4]])
        response = await client.client.post(
            "/predict_ndarray_multi_output",
            headers={"Content-Type": "application/json"},
            data=data,
        )
        assert response.status_code == 200
        assert await response.aread() == b"[[2, 4], [6, 8]]"

        data = json.dumps([1, 2, 3, 4])
        response = await client.client.post(
            "/predict_ndarray_enforce_shape",
            headers={"Content-Type": "application/json"},
            data=data,
        )
        assert response.status_code == 400

        data = json.dumps([[2, 1], [4, 3]])
        response = await client.client.post(
            "/predict_ndarray_enforce_dtype",
            headers={"Content-Type": "application/json"},
            data=data,
        )
        assert response.status_code == 200
        assert await response.aread() == b'[["4", "2"], ["8", "6"]]'

        data = json.dumps([["2f", 1], [4, 3]])
        response = await client.client.post(
            "/predict_ndarray_enforce_dtype",
            headers={"Content-Type": "application/json"},
            data=data,
        )
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_json(host: str):
    ORIGIN = "http://bentoml.ai"
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        headers = {"Content-Type": "application/json", "Origin": ORIGIN}
        data = json.dumps("hi")
        response = await client.client.post("/echo_json", headers=headers, data=data)
        assert response.status_code == 200
        assert await response.aread() == b'"hi"'

        headers = {"Content-Type": "application/json", "Origin": ORIGIN}
        data = json.dumps("hi")
        response = await client.client.post(
            "/echo_json_sync", headers=headers, data=data
        )
        assert response.status_code == 200
        assert await response.aread() == b'"hi"'

        headers = {"Content-Type": "application/json", "Origin": ORIGIN}
        data = json.dumps({"name": "test", "endpoints": ["predict", "health"]})
        response = await client.client.post(
            "/echo_json_enforce_structure", headers=headers, data=data
        )
        assert response.status_code == 200
        assert (
            await response.aread()
            == b'{"name":"test","endpoints":["predict","health"]}'
        )


@pytest.mark.asyncio
async def test_obj(host: str):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        for obj in [1, 2.2, "str", [1, 2, 3], {"a": 1, "b": 2}]:
            obj_str = json.dumps(obj, separators=(",", ":"))
            response = await client.client.post(
                "/echo_obj",
                headers={"Content-Type": "application/json"},
                data=obj_str,
            )
            assert response.status_code == 200
            assert await response.aread() == obj_str.encode("utf-8")


@pytest.mark.asyncio
async def test_pandas(host: str):
    import pandas as pd

    ORIGIN = "http://bentoml.ai"

    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        df = pd.DataFrame([[101]], columns=["col1"])

        headers = {"Content-Type": "application/json", "Origin": ORIGIN}
        data = df.to_json(orient="records")
        response = await client.client.post(
            "/predict_dataframe", headers=headers, data=data
        )
        assert response.status_code == 200
        assert await response.aread() == b'[{"col1":202}]'

        headers = {"Content-Type": "application/octet-stream", "Origin": ORIGIN}
        data = df.to_parquet()
        response = await client.client.post(
            "/predict_dataframe", headers=headers, data=data
        )
        assert response.status_code == 200
        assert await response.aread() == b'[{"col1":202}]'

        headers = {"Content-Type": "application/vnd.apache.parquet", "Origin": ORIGIN}
        data = df.to_parquet()
        response = await client.client.post(
            "/predict_dataframe", headers=headers, data=data
        )
        assert response.status_code == 200
        assert await response.aread() == b'[{"col1":202}]'

        headers = {"Content-Type": "text/csv", "Origin": ORIGIN}
        data = df.to_csv()
        response = await client.client.post(
            "/predict_dataframe", headers=headers, data=data
        )
        assert response.status_code == 200
        assert await response.aread() == b'[{"col1":202}]'


@pytest.mark.asyncio
async def test_file(host: str, bin_file: str):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        # Test File as binary
        with open(str(bin_file), "rb") as f:
            b = f.read()

        response = await client.client.post(
            "/predict_file",
            data=b,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert await response.aread() == b"\x810\x899"

        # Test File as multipart binary
        files = {"file": ("file.bin", b, "application/octet-stream")}
        response = await client.client.post("/predict_file", files=files)
        assert await response.aread() == b"\x810\x899"

        response = await client.client.post(
            "/predict_file",
            data=b,
            headers={"Content-Type": "application/pdf"},
        )
        assert await response.aread() == b"\x810\x899"


@pytest.mark.asyncio
async def test_image(host: str, img_file: str):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        with open(str(img_file), "rb") as f1:
            img_bytes = f1.read()

        response = await client.client.post(
            "/echo_image",
            data=img_bytes,
            headers={"Content-Type": "image/bmp"},
        )
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "image/bmp"

        bio = io.BytesIO(await response.aread())
        bio.name = "test.bmp"
        img = PILImage.open(bio)
        array1 = np.array(img)
        array2 = PILImage.open(img_file)

        np.testing.assert_array_almost_equal(array1, np.array(array2))

        response = await client.client.post(
            "/echo_image",
            data=img_bytes,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

        with open(str(img_file), "rb") as f1:
            b = f1.read()
        response = await client.client.post(
            "/echo_image",
            data=b,
            headers={"Content-Type": "application/pdf"},
        )
        assert response.status_code == 400


@pytest.fixture(name="img_form_data")
def fixture_img_form_data(img_file: str) -> Dict[str, Tuple[str, bytes, str]]:
    with open(img_file, "rb") as f1, open(img_file, "rb") as f2:
        files = {
            "original": (img_file, f1.read(), "image/bmp"),
            "compared": (img_file, f2.read(), "image/bmp"),
        }
    yield files


@pytest.mark.asyncio
async def test_multipart_image_io(
    host: str, img_form_data: Dict[str, Tuple[str, bytes, str]]
):
    from starlette.datastructures import UploadFile

    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        response = await client.client.post(
            "/predict_multi_images", files=img_form_data
        )
        assert response.status_code == 200

        form = await parse_multipart_form(
            headers=response.headers, body=response.content
        )
        for _, v in form.items():
            assert isinstance(v, UploadFile)
            img = PILImage.open(v.file)
            assert np.array(img).shape == (10, 10, 3)


@pytest.mark.asyncio
async def test_multipart_different_args(
    host: str, img_form_data: Dict[str, Tuple[str, bytes, str]]
):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        response = await client.client.post(
            "/predict_different_args", files=img_form_data
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_text_stream(host: str):
    async with await AsyncHTTPClient.from_url(f"http://{host}") as client:
        response = await client.client.post(
            "/predict_text_stream",
            headers={"Content-Type": "text/plain"},
            data="yo",
        )
        assert response.status_code == 200
        assert await response.aread() == b"yo 0yo 1yo 2yo 3yo 4yo 5yo 6yo 7yo 8yo 9"
