import time
import typing as t
import asyncio
import logging
import functools
import traceback
import collections

import numpy as np

from ..utils import cached_property
from ..utils.alg import TokenBucket

logger = logging.getLogger(__name__)


class NonBlockSema:
    def __init__(self, count):
        self.sema = count

    def acquire(self):
        if self.sema < 1:
            return False
        self.sema -= 1
        return True

    def is_locked(self):
        return self.sema < 1

    def release(self):
        self.sema += 1


class Optimizer:
    """
    Analyse historical data to optimize CorkDispatcher.
    """

    N_KEPT_SAMPLE = 50  # amount of outbound info kept for inferring params
    N_SKIPPED_SAMPLE = 2  # amount of outbound info skipped after init
    INTERVAL_REFRESH_PARAMS = 5  # seconds between each params refreshing

    def __init__(self):
        """
        assume the outbound duration follows duration = o_a * n + o_b
        (all in seconds)
        """
        self.o_stat = collections.deque(
            maxlen=self.N_KEPT_SAMPLE
        )  # to store outbound stat data
        self.o_a = 2
        self.o_b = 1

        self.wait = 0.01  # the avg wait time before outbound called

        self._refresh_tb = TokenBucket(2)  # to limit params refresh interval
        self._outbound_counter = 0

    def log_outbound(self, n, wait, duration):
        if (
            self._outbound_counter <= self.N_SKIPPED_SAMPLE
        ):  # skip inaccurate info at beginning
            self._outbound_counter += 1
            return

        self.o_stat.append((n, duration, wait))

        if self._refresh_tb.consume(1, 1.0 / self.INTERVAL_REFRESH_PARAMS, 1):
            self.trigger_refresh()

    def trigger_refresh(self):
        x = tuple((i, 1) for i, _, _ in self.o_stat)
        y = tuple(i for _, i, _ in self.o_stat)
        _o_a, _o_b = np.linalg.lstsq(x, y, rcond=None)[0]
        _o_w = sum(w for _, _, w in self.o_stat) * 1.0 / len(self.o_stat)

        self.o_a, self.o_b = max(0.000001, _o_a), max(0, _o_b)
        self.wait = max(0, _o_w)
        logger.debug(
            "Dynamic batching optimizer params updated: o_a: %.6f, o_b: %.6f, wait: %.6f",
            _o_a,
            _o_b,
            _o_w,
        )


T_IN = t.TypeVar("T_IN")
T_OUT = t.TypeVar("T_OUT")


class CorkDispatcher:
    """
    A decorator that:
        * wrap batch function
        * implement CORK algorithm to cork & release calling of wrapped function
    The wrapped function should be an async function.
    """

    def __init__(
        self,
        max_latency_in_ms: int,
        max_batch_size: int,
        shared_sema: t.Optional[NonBlockSema] = None,
        fallback: t.Optional[t.Callable[[], t.Any]] = None,
    ):
        """
        params:
            * max_latency_in_ms: max_latency_in_ms for inbound tasks in milliseconds
            * max_batch_size: max batch size of inbound tasks
            * shared_sema: semaphore to limit concurrent outbound tasks
            * fallback: callable to return fallback result
        raises:
            * all possible exceptions the decorated function has
        """
        self.max_latency_in_ms = max_latency_in_ms / 1000.0
        self.callback = None
        self.fallback = fallback
        self.optimizer = Optimizer()
        self.max_batch_size = int(max_batch_size)
        self.tick_interval = 0.001

        self._controller = None
        self._queue = collections.deque()  # TODO(hrmthw): maxlen
        self._sema = shared_sema if shared_sema else NonBlockSema(1)

    def shutdown(self):
        if self._controller is not None:
            self._controller.cancel()
        try:
            while True:
                _, _, fut = self._queue.pop()
                fut.cancel()
        except IndexError:
            pass

    @cached_property
    def _loop(self):
        return asyncio.get_event_loop()

    @cached_property
    def _wake_event(self):
        return asyncio.Condition()

    def __call__(
        self,
        callback: t.Callable[
            [t.Iterable[T_IN]], t.Coroutine[None, None, t.Iterable[T_OUT]]
        ],
    ) -> t.Callable[[T_IN], t.Coroutine[None, None, T_OUT]]:
        self.callback = callback

        @functools.wraps(callback)
        async def _func(data):
            if self._controller is None:
                self._controller = self._loop.create_task(self.controller())
            try:
                r = await self.inbound_call(data)
            except asyncio.CancelledError:
                return None if self.fallback is None else self.fallback()
            if isinstance(r, Exception):
                raise r
            return r

        return _func

    async def controller(self):
        """
        A standalone coroutine to wait/dispatch calling.
        """
        while True:
            try:
                async with self._wake_event:  # block until there's any request in queue
                    await self._wake_event.wait_for(self._queue.__len__)

                n = len(self._queue)
                dt = self.tick_interval
                decay = 0.95  # the decay rate of wait time
                now = time.time()
                w0 = now - self._queue[0][0]
                wn = now - self._queue[-1][0]
                a = self.optimizer.o_a
                b = self.optimizer.o_b

                if n > 1 and (w0 + a * n + b) >= self.max_latency_in_ms:
                    self._queue.popleft()[2].cancel()
                    continue
                if self._sema.is_locked():
                    if n == 1 and w0 >= self.max_latency_in_ms:
                        self._queue.popleft()[2].cancel()
                        continue
                    await asyncio.sleep(self.tick_interval)
                    continue
                if n * (wn + dt + (a or 0)) <= self.optimizer.wait * decay:
                    await asyncio.sleep(self.tick_interval)
                    continue

                n_call_out = min(
                    self.max_batch_size,
                    n,
                )
                # call
                self._sema.acquire()
                inputs_info = tuple(self._queue.pop() for _ in range(n_call_out))
                self._loop.create_task(self.outbound_call(inputs_info))
            except asyncio.CancelledError:
                break
            except Exception:  # pylint: disable=broad-except
                logger.error(traceback.format_exc())

    async def inbound_call(self, data):
        now = time.time()
        future = self._loop.create_future()
        input_info = (now, data, future)
        self._queue.append(input_info)
        async with self._wake_event:
            self._wake_event.notify_all()
        return await future

    async def outbound_call(self, inputs_info):
        _time_start = time.time()
        _done = False
        logger.debug("Dynamic batching cork released, batch size: %d", len(inputs_info))
        try:
            outputs = await self.callback(tuple(d for _, d, _ in inputs_info))
            assert len(outputs) == len(inputs_info)
            for (_, _, fut), out in zip(inputs_info, outputs):
                if not fut.done():
                    fut.set_result(out)
            _done = True
            self.optimizer.log_outbound(
                n=len(inputs_info),
                wait=_time_start - inputs_info[-1][0],
                duration=time.time() - _time_start,
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:  # pylint: disable=broad-except
            for _, _, fut in inputs_info:
                if not fut.done():
                    fut.set_result(e)
            _done = True
        finally:
            if not _done:
                for _, _, fut in inputs_info:
                    if not fut.done():
                        fut.cancel()
            self._sema.release()
