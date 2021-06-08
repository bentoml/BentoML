import asyncio

import psutil
import pytest

DEFAULT_MAX_LATENCY = 10 * 1000


@pytest.mark.skipif(not psutil.POSIX, reason="production server only works on POSIX")
@pytest.mark.asyncio
async def test_batch_size_limit(host):
    A, B = 0.0002, 0.01
    data = '{"a": %s, "b": %s}' % (A, B)

    # test for max_batch_size=None
    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_batch_size",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=lambda i: i in (200, 429),
        )
        for _ in range(100)
    )
    await asyncio.gather(*tasks)

    tasks = tuple(
        pytest.assert_request(
            "POST",
            f"http://{host}/echo_batch_size",
            headers=(("Content-Type", "application/json"),),
            data=data,
            assert_status=200,
            assert_data=lambda d: d == b'429: Too Many Requests'
            or int(d.decode()) == 1,
        )
        for _ in range(30)
    )
    await asyncio.gather(*tasks)
