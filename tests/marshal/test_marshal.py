import time
import asyncio

from bentoml.marshal.marshal import ParadeDispatcher


async def test_parade_dispatcher():
    MAX_LATENCY = 0.1

    def _test_func(i):
        return i * 2

    @ParadeDispatcher(MAX_LATENCY, max_size=1000)
    async def _do_sth_slow(input_list):
        await asyncio.sleep(0.1)
        return [_test_func(i) for i in input_list]

    inputs = [i for i in range(100)]
    time_st = time.time()
    outputs = await asyncio.gather(*(_do_sth_slow(i) for i in inputs))
    assert time.time() - time_st < 1
    assert [o == _test_func(i) for i, o in zip(inputs, outputs)]


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_parade_dispatcher())
