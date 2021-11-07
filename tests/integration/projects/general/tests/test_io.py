# pylint: disable=redefined-outer-name
import numpy as np
import pytest


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


import aiohttp


@pytest.mark.asyncio
async def test_file(host, bin_file, async_request):
    # Test FileInput as binary
    with open(str(bin_file), "rb") as f:
        b = f.read()

    await async_request(
        "POST",
        f"http://{host}/predict_file",
        data=b,
        headers={"Content-Type": "application/octet-stream"},
        assert_data=b"\x810\x899",
    )

    # Test FileInput as multipart binary
    form = aiohttp.FormData()
    form.add_field("file", b, content_type="application/octet-stream")

    await async_request(
        "POST",
        f"http://{host}/predict_file",
        data=form,
        assert_data=b"\x810\x899",
    )


import aiohttp
import imageio


@pytest.fixture()
def img_file(tmpdir):
    img_file_ = tmpdir.join("test_img.jpg")
    imageio.imwrite(str(img_file_), np.zeros((10, 10)))
    return str(img_file_)


@pytest.mark.asyncio
async def test_image(host, img_file, async_request):
    import imageio  # noqa # pylint: disable=unused-import
    import numpy as np  # noqa # pylint: disable=unused-import

    # Test MultiImageInput.
    with open(str(img_file), "rb") as f1:
        with open(str(img_file), "rb") as f2:
            form = aiohttp.FormData()
            form.add_field("original", f1.read(), content_type="image/jpeg")
            form.add_field("compared", f2.read(), content_type="image/jpeg")

            await async_request(
                "POST",
                f"http://{host}/predict_multi_images",
                data=form,
                assert_data=b"true",
            )
    """
    # Test ImageInput as binary
    with open(str(img_file), "rb") as f:
        img = f.read()
        await async_request(
            "POST", f"http://{host}/predict_image", data=img, assert_data=b"[10, 10, 3]"
        )

    # Test ImageInput as multipart binary
    with open(str(img_file), "rb") as f:
        await async_request(
            "POST",
            f"http://{host}/predict_image",
            data={"image": f},
            assert_data=b"[10, 10, 3]",
        )
    """


"""
@pytest.fixture(params=pytest.DF_AUTO_ORIENTS)
def df_orient(request):
    return request.param


@pytest.mark.asyncio
async def test_api_echo_json(host):
    for data in ('"hello"', '"🙂"', '"CJK汉语日本語한국어"'):
        await async_request(
            "POST",
            f"http://{host}/echo_json",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=200,
            assert_data=data.encode(),
        )


@pytest.since_bentoml_version("0.12.1+0", skip_by_default=True)
@pytest.mark.asyncio
async def test_api_echo_json_ensure_ascii(host):
    for data in ('"hello"', '"🙂"', '"CJK汉语日本語한국어"'):
        await async_request(
            "POST",
            f"http://{host}/echo_json_ensure_ascii",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=200,
            assert_data=json.dumps(json.loads(data)).encode(),
        )


@pytest.mark.asyncio
async def test_api_server_dataframe(host, df_orient):
    import pandas as pd  # noqa # pylint: disable=unused-import

    df = pd.DataFrame([[10], [20]], columns=["col1"])
    data = df.to_json(orient=df_orient)

    await async_request(
        "POST",
        f"http://{host}/predict_dataframe",
        headers=(("Content-Type", "application/json"),),
        data=data,
        assert_status=200,
        assert_data=lambda d: d.decode().strip() == '[{"col1":20},{"col1":40}]',
    )

    await async_request(
        "POST",
        f"http://{host}/predict_dataframe_v1",
        headers=(("Content-Type", "application/json"),),
        data=data,
        assert_status=200,
        assert_data=lambda d: d.decode().strip() == '[{"col1":20},{"col1":40}]',
    )


@pytest.mark.asyncio
async def test_api_server_image(host, img_file):
    import imageio  # noqa # pylint: disable=unused-import
    import numpy as np  # noqa # pylint: disable=unused-import

    # Test ImageInput as binary
    with open(str(img_file), "rb") as f:
        img = f.read()
        await async_request(
            "POST", f"http://{host}/predict_image", data=img, assert_data=b"[10, 10, 3]"
        )

    # Test ImageInput as multipart binary
    with open(str(img_file), "rb") as f:
        await async_request(
            "POST",
            f"http://{host}/predict_image",
            data={"image": f},
            assert_data=b"[10, 10, 3]",
        )

    # Test MultiImageInput.
    with open(str(img_file), "rb") as f1:
        with open(str(img_file), "rb") as f2:
            await async_request(
                "POST",
                f"http://{host}/predict_multi_images",
                data={"original": f1, "compared": f2},
                assert_data=b"true",
            )


@pytest.mark.asyncio
async def test_api_server_file(host, bin_file):
    # Test FileInput as binary
    with open(str(bin_file), "rb") as f:
        b = f.read()
        await async_request(
            "POST",
            f"http://{host}/predict_file",
            data=b,
            assert_data=b'{"b64": "gTCJOQ=="}',
        )

    # Test FileInput as multipart binary
    with open(str(bin_file), "rb") as f:
        await async_request(
            "POST",
            f"http://{host}/predict_file",
            data={"file": f},
            assert_data=b'{"b64": "gTCJOQ=="}',
        )


@pytest.mark.asyncio
async def test_api_server_json(host):
    req_count = 3
    tasks = tuple(
        async_request(
            "POST",
            f"http://{host}/predict_json",
            headers=(("Content-Type", "application/json"),),
            data=json.dumps({"in": i}),
            assert_data=bytes('{"in": %s}' % i, "ascii"),
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_api_server_tasks_api(host):
    req_count = 2
    tasks = tuple(
        async_request(
            "POST",
            f"http://{host}/predict_strict_json",
            headers=(("Content-Type", "application/json"),),
            data=json.dumps({"in": i}),
            assert_status=200,
            assert_data=bytes('{"in": %s}' % i, "ascii"),
        )
        for i in range(req_count)
    )
    tasks += tuple(
        async_request(
            "POST",
            f"http://{host}/predict_strict_json",
            data=json.dumps({"in": i}),
            assert_status=400,
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_api_server_inference_result(host):
    req_count = 2
    tasks = tuple(
        async_request(
            "POST",
            f"http://{host}/predict_direct_json",
            headers=(("Content-Type", "application/json"),),
            data=json.dumps({"in": i}),
            assert_status=200,
            assert_data=bytes('{"in": %s}' % i, "ascii"),
        )
        for i in range(req_count)
    )
    tasks += tuple(
        async_request(
            "POST",
            f"http://{host}/predict_direct_json",
            data=json.dumps({"in": i}),
            assert_status=400,
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)
"""
