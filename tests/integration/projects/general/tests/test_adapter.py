# pylint: disable=redefined-outer-name
import asyncio
import json

import pytest


@pytest.fixture(params=pytest.DF_AUTO_ORIENTS)
def df_orient(request):
    return request.param


@pytest.mark.asyncio
async def test_api_server_dataframe(host, df_orient):
    import pandas as pd  # noqa # pylint: disable=unused-import

    df = pd.DataFrame([[10], [20]], columns=['col1'])
    data = df.to_json(orient=df_orient)

    await pytest.assert_request(
        "POST",
        f"http://{host}/predict_dataframe",
        headers=(("Content-Type", "application/json"),),
        data=data,
        assert_status=200,
        assert_data=lambda d: d.decode().strip() == '[{"col1":20},{"col1":40}]',
    )

    await pytest.assert_request(
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
        await pytest.assert_request(
            "POST", f"http://{host}/predict_image", data=img, assert_data=b"[10, 10, 3]"
        )

    # Test ImageInput as multipart binary
    with open(str(img_file), "rb") as f:
        await pytest.assert_request(
            "POST",
            f"http://{host}/predict_image",
            data={"image": f},
            assert_data=b"[10, 10, 3]",
        )

    # Test MultiImageInput.
    with open(str(img_file), "rb") as f1:
        with open(str(img_file), "rb") as f2:
            await pytest.assert_request(
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
        await pytest.assert_request(
            "POST",
            f"http://{host}/predict_file",
            data=b,
            assert_data=b'{"b64": "gTCJOQ=="}',
        )

    # Test FileInput as multipart binary
    with open(str(bin_file), "rb") as f:
        await pytest.assert_request(
            "POST",
            f"http://{host}/predict_file",
            data={"file": f},
            assert_data=b'{"b64": "gTCJOQ=="}',
        )


@pytest.mark.asyncio
async def test_api_server_json(host):
    req_count = 3 if pytest.enable_microbatch else 1
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/predict_json",
            headers=(("Content-Type", "application/json"),),
            data=json.dumps({"in": i}),
            assert_data=bytes('{"in": %s}' % i, 'ascii'),
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_api_server_json_with_customized_route(host):
    req_count = 3 if pytest.enable_microbatch else 1
    tasks = tuple(
            pytest.assert_request(
                "POST",
                f"http://{host}/v1/predict_json",
                headers=(("Content-Type", "application/json"),),
                data=json.dumps({"in":i}),
                assert_data=bytes('{"in": %s}' % i, 'ascii'),
                )
            for i in range(req_count)
            )
    await asyncio.gather(*tasks)

@pytest.mark.asyncio
async def test_api_server_json_with_customized_route(host):
    req_count = 3 if pytest.enable_microbatch else 1
    tasks = tuple(
            pytest.assert_request(
                "POST",
                f"http://{host}/$~!@%^&*()_-+=[]\|;:,./predict_json", 
                headers=(("Content-Type", "application/json"),),
                data=json.dumps({"in":i}),
                assert_data=bytes('{"in": %s}' % i, 'ascii'),
                )
            for i in range(req_count)
            )
    await asyncio.gather(*tasks)

@pytest.mark.asyncio
async def test_api_server_tasks_api(host):
    req_count = 2 if pytest.enable_microbatch else 1
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/predict_strict_json",
            headers=(("Content-Type", "application/json"),),
            data=json.dumps({"in": i}),
            assert_status=200,
            assert_data=bytes('{"in": %s}' % i, 'ascii'),
        )
        for i in range(req_count)
    )
    tasks += tuple(
        pytest.assert_request(
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
    req_count = 2 if pytest.enable_microbatch else 1
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/predict_direct_json",
            headers=(("Content-Type", "application/json"),),
            data=json.dumps({"in": i}),
            assert_status=200,
            assert_data=bytes('{"in": %s}' % i, 'ascii'),
        )
        for i in range(req_count)
    )
    tasks += tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/predict_direct_json",
            data=json.dumps({"in": i}),
            assert_status=400,
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)
