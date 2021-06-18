# pylint: disable=redefined-outer-name
import time

import pytest


@pytest.mark.asyncio
async def test_SLO(host):
    await pytest.assert_request(
        "POST",
        f"http://{host}/echo_with_delay_max3",
        data='"0"',
        headers=(("Content-Type", "application/json"),),
        assert_status=200,
    )

    SLO = 3
    accuracy = 0.01

    time_start = time.time()
    await pytest.assert_request(
        "POST",
        f"http://{host}/echo_with_delay_max3",
        data='"2.9"',
        timeout=SLO * 2,
        headers=(("Content-Type", "application/json"),),
        assert_status=200,
    )
    assert time.time() - time_start < SLO * (1 + accuracy)

    time_start = time.time()
    await pytest.assert_request(
        "POST",
        f"http://{host}/echo_with_delay_max3",
        data='"3.5"',
        timeout=SLO * 2,
        headers=(("Content-Type", "application/json"),),
        assert_status=408,
    )
    assert time.time() - time_start < SLO * (1 + accuracy)
