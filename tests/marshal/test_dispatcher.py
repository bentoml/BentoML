import pytest
import time
import asyncio
from bentoml.marshal.dispatcher import CorkDispatcher


@pytest.mark.asyncio
async def test_dispatcher_raise_error():
    @CorkDispatcher(max_batch_size=10, max_latency_in_ms=100)
    async def f(xs):
        for x in xs:
            if x == 1:
                raise ValueError()
            if x == 2:
                raise TypeError()
        return xs

    with pytest.raises(ValueError):
        await f(1)

    with pytest.raises(TypeError):
        await f(2)
    assert await f(3) == 3


@pytest.mark.asyncio
async def test_dispatcher():
    MAX_LATENCY = 0.1

    def _test_func(i):
        return i * 2

    @CorkDispatcher(MAX_LATENCY, 1000)
    async def _do_sth_slow(input_list):
        await asyncio.sleep(0.1)
        return [_test_func(i) for i in input_list]

    inputs = [i for i in range(100)]
    time_st = time.time()
    outputs = await asyncio.gather(*(_do_sth_slow(i) for i in inputs))
    assert time.time() - time_st < 1
    assert [o == _test_func(i) for i, o in zip(inputs, outputs)]
