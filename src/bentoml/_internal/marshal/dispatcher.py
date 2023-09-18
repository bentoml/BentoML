from __future__ import annotations

import time
import typing as t
import asyncio
import logging
import functools
import traceback
import collections

import attr
import numpy as np

from ..utils import cached_property
from ..utils.alg import TokenBucket

logger = logging.getLogger(__name__)


if t.TYPE_CHECKING:
    from ..runner.utils import Params
    from ..runner.container import Payload


class NonBlockSema:
    def __init__(self, count: int):
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


@attr.define
class Job:
    enqueue_time: float
    data: Params[Payload]
    future: asyncio.Future[t.Any]
    dispatch_time: float = 0


class Optimizer:
    """
    Analyse historical data to optimize CorkDispatcher.
    """

    N_KEPT_SAMPLE = 50  # amount of outbound info kept for inferring params
    N_SKIPPED_SAMPLE = 2  # amount of outbound info skipped after init
    INTERVAL_REFRESH_PARAMS = 5  # seconds between each params refreshing

    def __init__(self, max_latency: float):
        """
        assume the outbound duration follows duration = o_a * n + o_b
        (all in seconds)
        """
        self.o_stat: collections.deque[tuple[int, float, float]] = collections.deque(
            maxlen=self.N_KEPT_SAMPLE
        )  # to store outbound stat data
        self.o_a = min(2, max_latency * 2.0 / 30)
        self.o_b = min(1, max_latency * 1.0 / 30)

        self.wait = 0  # the avg wait time before outbound called

        self._refresh_tb = TokenBucket(2)  # to limit params refresh interval
        self.outbound_counter = 0

    def log_outbound(self, n: int, wait: float, duration: float):
        if self.outbound_counter <= self.N_SKIPPED_SAMPLE + 4:
            self.outbound_counter += 1
            # skip inaccurate info at beginning
            if self.outbound_counter <= self.N_SKIPPED_SAMPLE:
                return

        self.o_stat.append((n, duration, wait))

        if self._refresh_tb.consume(1, 1.0 / self.INTERVAL_REFRESH_PARAMS, 1):
            self.trigger_refresh()

    def trigger_refresh(self):
        if not self.o_stat:
            logger.debug(
                "o_stat is empty, skip dynamic batching optimizer params update"
            )
            return

        x = tuple((i, 1) for i, _, _ in self.o_stat)
        y = tuple(i for _, i, _ in self.o_stat)

        _factors: tuple[float, float] = np.linalg.lstsq(x, y, rcond=None)[0]  # type: ignore
        _o_a, _o_b = _factors
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
        fallback: t.Callable[[], t.Any] | type[t.Any] | None = None,
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
        self.max_latency = max_latency_in_ms / 1000.0
        self.fallback = fallback
        self.optimizer = Optimizer(self.max_latency)
        self.max_batch_size = int(max_batch_size)
        self.tick_interval = 0.001

        self._controller = None
        self._queue: collections.deque[
            Job
        ] = collections.deque()  # TODO(bojiang): maxlen
        self._sema = shared_sema if shared_sema else NonBlockSema(1)

    def shutdown(self):
        if self._controller is not None:
            self._controller.cancel()
        for job in self._queue:
            job.future.cancel()

    @cached_property
    def _loop(self):
        return asyncio.get_event_loop()

    @cached_property
    def _wake_event(self):
        return asyncio.Condition()

    def __call__(
        self,
        callback: t.Callable[
            [t.Sequence[T_IN]], t.Coroutine[None, None, t.Sequence[T_OUT]]
        ],
    ) -> t.Callable[[T_IN], t.Coroutine[None, None, T_OUT]]:
        self.callback = callback

        @functools.wraps(callback)
        async def _func(data: t.Any) -> t.Any:
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

    async def train_optimizer(
        self,
        num_required_reqs: int,
        num_reqs_to_train: int,
        batch_size: int,
    ):
        if self.max_batch_size < batch_size:
            batch_size = self.max_batch_size

        if batch_size > 1:
            wait = min(
                self.max_latency * 0.95,
                (batch_size * 2 + 1) * (self.optimizer.o_a + self.optimizer.o_b),
            )

        req_count = 0
        try:
            while req_count < num_reqs_to_train:
                async with self._wake_event:  # block until there's any request in queue
                    await self._wake_event.wait_for(self._queue.__len__)

                n = len(self._queue)
                now = time.time()
                w0 = now - self._queue[0].enqueue_time

                # only cancel requests if there are more than enough for training
                if n > num_required_reqs - req_count and w0 >= self.max_latency:
                    # we're being very conservative and only canceling requests if they have already timed out
                    self._queue.popleft().future.cancel()
                    continue
                if batch_size > 1:  # only wait if batch_size
                    a = self.optimizer.o_a
                    b = self.optimizer.o_b

                    if n < batch_size and (batch_size * a + b) + w0 <= wait:
                        await asyncio.sleep(self.tick_interval)
                        continue
                if self._sema.is_locked():
                    await asyncio.sleep(self.tick_interval)
                    continue

                n_call_out = min(n, batch_size)
                req_count += 1
                # call
                self._sema.acquire()
                inputs_info = tuple(self._queue.pop() for _ in range(n_call_out))
                for info in inputs_info:
                    # fake wait as 0 for training requests
                    info.enqueue_time = now
                self._loop.create_task(self.outbound_call(inputs_info))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logger.error(traceback.format_exc(), exc_info=e)

    async def controller(self):
        """
        A standalone coroutine to wait/dispatch calling.
        """
        try:
            logger.debug("Starting dispatcher optimizer training...")
            # warm up the model
            await self.train_optimizer(
                self.optimizer.N_SKIPPED_SAMPLE, self.optimizer.N_SKIPPED_SAMPLE + 6, 1
            )

            logger.debug("Dispatcher finished warming up model.")

            await self.train_optimizer(6, 1, 1)
            self.optimizer.trigger_refresh()
            logger.debug("Dispatcher finished optimizer training request 1.")

            await self.train_optimizer(5, 1, 2)
            self.optimizer.trigger_refresh()
            logger.debug("Dispatcher finished optimizer training request 2.")

            await self.train_optimizer(3, 1, 3)
            self.optimizer.trigger_refresh()
            logger.debug("Dispatcher finished optimizer training request 3.")

            if self.optimizer.o_a + self.optimizer.o_b >= self.max_latency:
                logger.warning(
                    "BentoML has detected that a service has a max latency that is likely too low for serving. If many 503 errors are encountered, try raising the 'runner.max_latency' in your BentoML configuration YAML file."
                )
            logger.debug("Dispatcher optimizer training complete.")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logger.error(traceback.format_exc(), exc_info=e)

        while True:
            try:
                async with self._wake_event:  # block until there's any request in queue
                    await self._wake_event.wait_for(self._queue.__len__)

                n = len(self._queue)
                dt = self.tick_interval
                decay = 0.95  # the decay rate of wait time
                now = time.time()
                w0 = now - self._queue[0].enqueue_time
                wn = now - self._queue[-1].enqueue_time
                a = self.optimizer.o_a
                b = self.optimizer.o_b

                # the estimated latency of the first request if we began processing now
                latency_0 = w0 + a * n + b

                if n > 1 and latency_0 >= self.max_latency:
                    self._queue.popleft().future.cancel()
                    continue
                if self._sema.is_locked():
                    if n == 1 and w0 >= self.max_latency:
                        self._queue.popleft().future.cancel()
                        continue
                    await asyncio.sleep(self.tick_interval)
                    continue
                if (
                    n < self.max_batch_size
                    and n * (wn + dt + (a or 0)) <= self.optimizer.wait * decay
                ):
                    await asyncio.sleep(self.tick_interval)
                    continue

                n_call_out = min(self.max_batch_size, n)
                # call
                self._sema.acquire()
                inputs_info = tuple(self._queue.pop() for _ in range(n_call_out))
                self._loop.create_task(self.outbound_call(inputs_info))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # pylint: disable=broad-except
                logger.error(traceback.format_exc(), exc_info=e)

    async def inbound_call(self, data: Params[Payload]):
        if data.sample.batch_size > self.max_batch_size:
            raise RuntimeError(
                f"batch of size {data.sample.batch_size} exceeds configured max batch size of {self.max_batch_size}."
            )

        now = time.time()
        future = self._loop.create_future()
        input_info = Job(now, data, future)
        self._queue.append(input_info)
        async with self._wake_event:
            self._wake_event.notify_all()
        return await future

    async def outbound_call(self, inputs_info: tuple[Job, ...]):
        _time_start = time.time()
        _done = False
        batch_size = len(inputs_info)
        logger.debug("Dynamic batching cork released, batch size: %d", batch_size)
        try:
            outputs = await self.callback(
                tuple(t.cast(t.Any, input_info.data) for input_info in inputs_info)
            )
            assert len(outputs) == len(inputs_info)
            for input_info, out in zip(inputs_info, outputs):
                fut = input_info.future
                if not fut.done():
                    fut.set_result(out)
            _done = True
            self.optimizer.log_outbound(
                n=len(inputs_info),
                wait=_time_start - inputs_info[-1].enqueue_time,
                duration=time.time() - _time_start,
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:  # pylint: disable=broad-except
            for input_info in inputs_info:
                fut = input_info.future
                if not fut.done():
                    fut.set_result(e)
            _done = True
        finally:
            if not _done:
                for input_info in inputs_info:
                    fut = input_info.future
                    if not fut.done():
                        fut.cancel()
            self._sema.release()
