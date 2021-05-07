# pylint: disable=redefined-outer-name

import asyncio
import time

import psutil  # noqa # pylint: disable=unused-import
import pytest

DEFAULT_MAX_LATENCY = 10 * 1000


@pytest.mark.skipif('not psutil.POSIX')
@pytest.mark.asyncio
async def test_slow_server(host):
    if not pytest.enable_microbatch:
        pytest.skip()

    A, B = 0.2, 1
    data = '{"a": %s, "b": %s}' % (A, B)

    time_start = time.time()
    req_count = 10
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_with_delay",
            headers=(("Content-Type", "application/json"),),
            data=data,
            timeout=30,
            assert_status=200,
            assert_data=data.encode(),
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)
    assert time.time() - time_start < 12


@pytest.mark.skipif('not psutil.POSIX')
@pytest.mark.asyncio
async def test_fast_server(host):
    if not pytest.enable_microbatch:
        pytest.skip()

    A, B = 0.0002, 0.01
    data = '{"a": %s, "b": %s}' % (A, B)

    req_count = 100
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_with_delay",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=lambda i: i in (200, 429),
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)

    time_start = time.time()
    req_count = 200
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_with_delay",
            headers=(("Content-Type", "application/json"),),
            data=data,
            timeout=30,
            assert_status=200,
            assert_data=data.encode(),
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)
    assert time.time() - time_start < 2
