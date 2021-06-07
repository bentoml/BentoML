import asyncio
import time

import psutil
import pytest

DEFAULT_MAX_LATENCY = 10 * 1000


@pytest.mark.skipif(not psutil.POSIX, reason="production server only works on POSIX")
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


@pytest.mark.skipif(not psutil.POSIX, reason="production server only works on POSIX")
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


@pytest.mark.skipif(not psutil.POSIX, reason="production server only works on POSIX")
@pytest.mark.asyncio
async def test_batch_size_limit(host):
    if not pytest.enable_microbatch:
        pytest.skip()

    A, B = 0.0002, 0.01
    data = '{"a": %s, "b": %s}' % (A, B)

    # test for max_batch_size=1
    req_count = 100
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_batch_size_1",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=lambda i: i in (200, 429),
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)

    req_count = 30
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_batch_size_1",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=200,
            assert_data=b"1",
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)

    # test for max_batch_size=None
    req_count = 100
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_batch_size",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=lambda i: i in (200, 429),
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)

    req_count = 30
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_batch_size",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=200,
            assert_data=lambda d: int(d.decode()) > 1,
        )
        for i in range(req_count)
    )
    await asyncio.gather(*tasks)
