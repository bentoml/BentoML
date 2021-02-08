# pylint: disable=redefined-outer-name

import pytest


@pytest.mark.asyncio
async def test_api_server_with_sklearn(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/predict_with_sklearn",
        headers=(("Content-Type", "application/json"),),
        data="[2.0]",
        assert_status=200,
        assert_data=b'2.0',
    )
