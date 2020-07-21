# pylint: disable=redefined-outer-name
import contextlib
import pytest
import time
import asyncio

import psutil  # noqa # pylint: disable=unused-import
from bentoml.marshal.dispatcher import CorkDispatcher


class BENCHMARK:
    N_PREHEAT_01 = 4
    N_PREHEAT_90 = 1000


@contextlib.contextmanager
def assert_in_time(max_time):
    start_time = time.time()
    yield
    time_elapsed = time.time() - start_time
    assert time_elapsed < max_time


def like(base, compared, percentage, measurement_error):
    compared = compared or 0
    error = abs(base - compared)
    allowed_error = percentage * base + measurement_error
    return error < allowed_error


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


@pytest.fixture(params=[(0.004, 0.01), (0.005, 0.005)])
def model(request):
    A, B = request.param

    class _Model:
        def __init__(self, A, B):
            self.A = A
            self.B = B
            self.n_called = 0

        async def __call__(self, input_list):
            self.n_called += 1
            await asyncio.sleep(self.B + self.A * len(input_list))
            return [i * 2 for i in input_list]

    return _Model(A, B)


@pytest.mark.asyncio
async def test_dispatcher_basic(model):
    N = 100
    dispatcher = CorkDispatcher(max_latency_in_ms=5 * 60 * 1000, max_batch_size=1000)

    A, B = model.A, model.B
    wrapped_model = dispatcher(model)

    inputs = [i for i in range(N)]

    with assert_in_time((A + B) * N):
        outputs = await asyncio.gather(*(wrapped_model(i) for i in inputs))
    assert model.n_called < N
    assert all([o == p for p, o in zip(await model(inputs), outputs)])


@pytest.mark.skipif('not psutil.POSIX')
@pytest.mark.asyncio
async def test_dispatcher_preheating(model):
    dispatcher = CorkDispatcher(max_latency_in_ms=2000, max_batch_size=1000)

    A, B = model.A, model.B
    wrapped_model = dispatcher(model)

    while model.n_called < BENCHMARK.N_PREHEAT_01 * 2:
        if like(B, dispatcher.optimizer.o_b, 0.99, 0.001):
            break
        await asyncio.gather(*(wrapped_model(i) for i in range(model.n_called % 5 + 1)))
        await asyncio.sleep(0.01)
    assert model.n_called <= BENCHMARK.N_PREHEAT_01

    while model.n_called < BENCHMARK.N_PREHEAT_90 * 2:
        if like(B, dispatcher.optimizer.o_b, 0.1, 0.001) and like(
            A, dispatcher.optimizer.o_a, 0.1, 0.001
        ):
            break
        await asyncio.gather(*(wrapped_model(i) for i in range(model.n_called % 5 + 1)))
        await asyncio.sleep(0.01)
    assert model.n_called <= BENCHMARK.N_PREHEAT_90


@pytest.mark.skipif('not psutil.POSIX')
@pytest.mark.asyncio
async def test_dispatcher_overload(model):
    dispatcher = CorkDispatcher(max_latency_in_ms=2000, max_batch_size=1000)

    wrapped_model = dispatcher(model)

    # preheating
    while model.n_called < BENCHMARK.N_PREHEAT_01:
        await asyncio.gather(*(wrapped_model(i) for i in range(5)))

    # check latency
    inputs = tuple(range(3000))
    with assert_in_time(2 * 1.5):
        outputs = await asyncio.gather(*(wrapped_model(i) for i in inputs))
    assert any([p == o for p, o in zip(await model(inputs), outputs)])
