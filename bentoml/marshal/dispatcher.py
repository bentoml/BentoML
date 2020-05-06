import asyncio
import logging
import traceback
import time
import collections
from typing import Callable
import numpy as np

from bentoml.utils.alg import FixedBucket, TokenBucket


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class NonBlockSema:
    def __init__(self, count):
        self.sema = count

    def aquire(self):
        if self.sema < 1:
            return False
        self.sema -= 1
        return True

    def is_locked(self):
        return self.sema < 1

    def release(self):
        self.sema += 1


class Optimizer:
    N_OUTBOUND_SAMPLE = 50
    INTERVAL_REFRESH_PARAMS = 5
    N_DATA_DROP_FIRST = 2

    def __init__(self):
        self.o_stat = FixedBucket(self.N_OUTBOUND_SAMPLE)
        self.o_a = 2
        self.o_b = 0.1
        self.o_w = 0.01
        self._refresh_tb = TokenBucket(2)
        self._outbound_init_counter = 0
        self._o_a = self.o_a
        self._o_b = self.o_b
        self._o_w = self.o_w

    def log_outbound(self, n, wait, duration):
        # drop first N_DATA_DROP_FIRST datas
        if self._outbound_init_counter <= self.N_DATA_DROP_FIRST:
            self._outbound_init_counter += 1
            return

        self.o_stat.put((n, duration, wait))

        if self._refresh_tb.consume(1, 1.0 / self.INTERVAL_REFRESH_PARAMS, 1):
            self.trigger_refresh()

    def trigger_refresh(self):
        x = tuple((i, 1) for i, _, _ in self.o_stat.data)
        y = tuple(i for _, i, _ in self.o_stat.data)
        self._o_a, self._o_b = np.linalg.lstsq(x, y, rcond=None)[0]
        self._o_w = sum(w for _, _, w in self.o_stat) * 1.0 / len(self.o_stat)

        self.o_a, self.o_b = max(0.000001, self._o_a), max(0, self._o_b)
        self.o_w = max(0, self._o_w)
        logger.info(
            "optimizer params updated: o_a: %.6f, o_b: %.6f, o_w: %.6f",
            self._o_a,
            self._o_b,
            self._o_w,
        )


class ParadeDispatcher:
    def __init__(
        self,
        max_expected_time: int,
        max_batch_size: int,
        shared_sema: NonBlockSema = None,
        fallback: Callable = None,
    ):
        """
        params:
            * max_expected_time: max_expected_time for inbound tasks in milliseconds
            * max_batch_size: max batch size of inbound tasks
            * shared_sema: semaphore to limit concurrent tasks
            * fallback: callable to return fallback result
        """
        self.max_expected_time = max_expected_time / 1000.0
        self.callback = None
        self.fallback = fallback
        self.optimizer = Optimizer()

        self.max_batch_size = int(max_batch_size)
        self._controller = None
        self._queue = collections.deque()  # TODO(hrmthw): maxlen
        self._loop = asyncio.get_event_loop()
        self._wake_event = asyncio.Condition()
        self._sema = shared_sema if shared_sema else NonBlockSema(1)

        self.tick_interval = 0.001

    def __call__(self, callback):
        self.callback = callback
        self._controller = self._loop.create_task(self.controller())

        async def _func(data):
            try:
                r = await self.inbound_call(data)
            except asyncio.CancelledError:
                return None if self.fallback is None else self.fallback()
            return r

        return _func

    async def controller(self):
        while True:
            try:
                async with self._wake_event:  # block until request in queue
                    await self._wake_event.wait_for(self._queue.__len__)

                n = len(self._queue)
                dt = self.tick_interval
                decay = 0.95
                now = time.time()
                w0 = now - self._queue[0][0]
                wn = now - self._queue[-1][0]
                a = self.optimizer.o_a
                b = self.optimizer.o_b

                if n > 1 and (w0 + a * n + b) >= self.max_expected_time:
                    self._queue.popleft()[2].cancel()
                    continue
                if self._sema.is_locked():
                    if n == 1 and w0 >= self.max_expected_time:
                        self._queue.popleft()[2].cancel()
                        continue
                    await asyncio.sleep(self.tick_interval)
                    continue
                if n * (wn + dt + a) <= self.optimizer.o_w * decay:
                    await asyncio.sleep(self.tick_interval)
                    continue

                n_call_out = min(self.max_batch_size, n,)
                # call
                self._sema.aquire()
                inputs_info = tuple(self._queue.pop() for _ in range(n_call_out))
                self._loop.create_task(self.outbound_call(inputs_info))
            except asyncio.CancelledError:
                break
            except Exception:  # pylint: disable=broad-except
                logger.error(traceback.format_exc())

    async def inbound_call(self, data) -> asyncio.Future:
        t = time.time()
        future = self._loop.create_future()
        input_info = (t, data, future)
        self._queue.append(input_info)
        async with self._wake_event:
            self._wake_event.notify_all()
        return await future

    async def outbound_call(self, inputs_info):
        _time_start = time.time()
        _done = False
        logger.info("outbound function called: %d", len(inputs_info))
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
        except Exception:  # pylint: disable=broad-except
            logger.error(traceback.format_exc())
        finally:
            if not _done:
                for _, _, fut in inputs_info:
                    if not fut.done():
                        fut.cancel()
            self._sema.release()
