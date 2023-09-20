from __future__ import annotations

import asyncio
import collections
import functools
import logging
import time
import traceback
import typing as t
from functools import cached_property
from abc import ABC
from abc import abstractmethod

import attr
import numpy as np

from ..utils import cached_property
from ..utils.alg import TokenBucket
from ...exceptions import BadInput

logger = logging.getLogger(__name__)


if t.TYPE_CHECKING:
    from ..runner.container import Payload
    from ..runner.utils import Params


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


OPTIMIZER_REGISTRY = {}


class Optimizer(ABC):
    optimizer_id: str
    n_skipped_sample: int = 0

    def __init__(self, options: dict[str, t.Any]):
        pass

    @abstractmethod
    def log_outbound(self, batch_size: int, duration: float):
        pass

    @abstractmethod
    def predict(self, batch_size: int) -> float:
        pass

    def predict_diff(self, first_batch_size: int, second_batch_size: int) -> float:
        """
        Predict the difference
        """
        return self.predict(second_batch_size) - self.predict(first_batch_size)

    def trigger_refresh(self):
        pass

    def __init_subclass__(cls, optimizer_id: str):
        OPTIMIZER_REGISTRY[optimizer_id] = cls
        cls.optimizer_id = optimizer_id


class FixedOptimizer(Optimizer, optimizer_id="fixed"):
    time: float

    def __init__(self, options: dict[str, t.Any]):
        if "time_ms" not in options:
            raise BadInput("Attempted to initialize ")
        self.time = options["time_ms"]

    def predict(self, batch_size: int):
        return self.time


class LinearOptimizer(Optimizer, optimizer_id="linear"):
    """
    Analyze historical data to predict execution time using a simple linear regression on batch size.
    """

    o_a: float = 2.0
    o_b: float = 1.0

    n_kept_sample = 50  # amount of outbound info kept for inferring params
    n_skipped_sample = 2  # amount of outbound info skipped after init
    param_refresh_interval = 5  # seconds between each params refreshing

    def __init__(self, options: dict[str, t.Any]):
        """
        assume the outbound duration follows duration = o_a * n + o_b
        (all in seconds)
        """
        for key in options:
            if key == "initial_slope_ms":
                self.o_a = options[key] / 1000.0
            elif key == "initial_intercept_ms":
                self.o_b = options[key] / 1000.0
            elif key == "n_kept_sample":
                self.n_kept_sample = options[key]
            elif key == "n_skipped_sample":
                self.n_skipped_sample = options[key]
            elif key == "param_refresh_interval":
                self.param_refresh_interval = options[key]
            else:
                logger.warning(
                    f"Optimizer 'linear' ignoring unknown configuration key '{key}'."
                )

        self.o_stat: collections.deque[tuple[int, float]] = collections.deque(
            maxlen=self.n_kept_sample
        )  # to store outbound stat data

        self._refresh_tb = TokenBucket(2)  # to limit params refresh interval
        self.outbound_counter = 0

    def log_outbound(self, batch_size: int, duration: float):
        if self.outbound_counter <= self.n_skipped_sample:
            # skip inaccurate info at beginning
            self.outbound_counter += 1
            return

        self.o_stat.append((batch_size, duration))

        if self._refresh_tb.consume(1, 1.0 / self.param_refresh_interval, 1):
            self.trigger_refresh()

    def predict(self, batch_size: int):
        return self.o_a * batch_size + self.o_b

    def predict_diff(self, first_batch_size: int, second_batch_size: int):
        return self.o_a * (second_batch_size - first_batch_size)

    def trigger_refresh(self):
        x = tuple((i, 1) for i, _ in self.o_stat)
        y = tuple(i for _, i in self.o_stat)

        _factors = t.cast(tuple[float, float], np.linalg.lstsq(x, y, rcond=None)[0])
        _o_a, _o_b = _factors

        self.o_a, self.o_b = max(0.000001, _o_a), max(0, _o_b)
        logger.debug(
            "Dynamic batching optimizer params updated: o_a: %.6f, o_b: %.6f",
            _o_a,
            _o_b,
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
        optimizer: Optimizer,
        strategy: BatchingStrategy,
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
        self.optimizer = optimizer
        self.strategy = strategy
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
        for task in self.background_tasks:
            task.cancel()
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

        wait = 0
        if batch_size > 1:
            wait = min(
                self.max_latency * 0.95,
                self.optimizer.predict(batch_size * 2 + 1),
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
                    if (
                        n < batch_size
                        and self.optimizer.predict(batch_size) + w0 <= wait
                    ):
                        await asyncio.sleep(self.tick_interval)
                        continue
                if self._sema.is_locked():
                    await asyncio.sleep(self.tick_interval)
                    continue

                if self.max_batch_size == -1:  # batching is disabled
                    n_call_out = 1
                    batch_size = self._queue[0].data.sample.batch_size
                else:
                    n_call_out = 0
                    batch_size = 0
                    for job in queue:
                        if batch_size + job.data.sample.batch_size <= max_batch_size:
                            n_call_out += 1
                            batch_size += job.data.sample.batch_size

                req_count += 1
                # call
                inputs_info = tuple(self._queue.pop() for _ in range(n_call_out))
                dispatch(inputs_info, batch_size)
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
                self.optimizer.n_skipped_sample, self.optimizer.n_skipped_sample + 6, 1
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

            if self.optimizer.predict(1) >= self.max_latency:
                logger.warning(
                    "BentoML has detected that a service has a max latency that is likely too low for serving. If many 503 errors are encountered, try raising the 'runner.max_latency' in your BentoML configuration YAML file."
                )
            logger.debug("Dispatcher optimizer training complete.")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(traceback.format_exc(), exc_info=e)

        while True:
            try:
                async with self._wake_event:  # block until there's any request in queue
                    await self._wake_event.wait_for(self._queue.__len__)

                n = len(self._queue)
                now = time.time()
                w0 = now - self._queue[0].enqueue_time

                # the estimated latency of the first request if we began processing now
                latency_0 = w0 + self.optimizer.predict(n)

                if n > 1 and latency_0 >= self.max_latency:
                    self._queue.popleft().future.cancel()
                    continue
                if self._sema.is_locked():
                    if n == 1 and w0 >= self.max_latency:
                        self._queue.popleft().future.cancel()
                        continue
                    await asyncio.sleep(self.tick_interval)
                    continue

                # we are now free to dispatch whenever we like
                if self.max_batch_size == -1:  # batching is disabled
                    n_call_out = 1
                    batch_size = self._queue[0].data.sample.batch_size
                else:
                    await self.strategy.batch(
                        self.optimizer,
                        self._queue,
                        self.max_latency,
                        self.max_batch_size,
                        self.tick_interval,
                        self._dispatch,
                    )

            except Exception as e:  # pylint: disable=broad-except
                logger.error(traceback.format_exc(), exc_info=e)

    def _dispatch(self, inputs_info: t.Sequence[Job], batch_size: int):
        self._sema.acquire()
        task = self._loop.create_task(self.outbound_call(inputs_info, batch_size))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def inbound_call(self, data: Params[Payload]):
        if self.max_batch_size > 0 and data.sample.batch_size > self.max_batch_size:
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

    async def outbound_call(self, inputs_info: t.Sequence[Job], batch_size: int):
        _time_start = time.time()
        _done = False
        logger.debug(
            "Dynamic batching cork released, batch size: %d (%d requests)",
            batch_size,
            len(inputs_info),
        )
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
                batch_size=len(inputs_info),
                duration=time.time() - _time_start,
            )
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
