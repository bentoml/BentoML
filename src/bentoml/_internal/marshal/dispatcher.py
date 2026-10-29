from __future__ import annotations

import asyncio
import collections
import functools
import heapq
import logging
import time
import typing as t
from functools import cached_property

import attr
import numpy as np
from starlette.concurrency import run_in_threadpool

from ..utils import is_async_callable
from ..utils.alg import TokenBucket

logger = logging.getLogger(__name__)


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


@attr.define
class Job(t.Generic[T_IN, T_OUT]):
    enqueue_time: float
    data: T_IN
    future: asyncio.Future[T_OUT | Exception]
    deadline: float
    dispatch_time: float = 0


@attr.frozen
class CandidateInfo:
    job_count: int
    sample_size: int
    first_enqueue_time: float
    last_enqueue_time: float
    earliest_deadline: float


@attr.frozen
class ScheduleDecision:
    dispatch: bool
    wait_time: float


def decide_next_action(
    *,
    now: float,
    candidate: CandidateInfo,
    predicted_duration: float,
    adaptive_should_wait: bool,
    semaphore_locked: bool,
    next_expiry: float | None,
    ticket_interval: float,
) -> ScheduleDecision:
    if semaphore_locked:
        wake_at = now + ticket_interval
        if next_expiry is not None:
            wake_at = min(wake_at, next_expiry)
        return ScheduleDecision(False, max(0.0, wake_at - now))

    latest_safe_start = candidate.earliest_deadline - predicted_duration
    if now >= latest_safe_start or not adaptive_should_wait:
        return ScheduleDecision(True, 0.0)

    wake_at = min(now + ticket_interval, latest_safe_start)
    if next_expiry is not None:
        wake_at = min(wake_at, next_expiry)
    return ScheduleDecision(False, max(0.0, wake_at - now))


class CorkDispatcher(t.Generic[T_IN, T_OUT]):
    """
    A decorator that:
        * wrap batch function
        * implement CORK algorithm to cork & release calling of wrapped function
    The wrapped function should be an async function.
    """

    callback: t.Callable[[t.Sequence[T_IN]], t.Coroutine[None, None, t.Sequence[T_OUT]]]
    # the interval between each poll
    TICKET_INTERVAL = 0.001

    def __init__(
        self,
        max_latency_in_ms: int,
        max_batch_size: int,
        *,
        shared_sema: t.Optional[NonBlockSema] = None,
        fallback: t.Callable[[], T_OUT],
        get_batch_size: t.Callable[[T_IN], int] = lambda x: x.sample.batch_size,
        batch_dim: tuple[int, int] = (0, 0),
        clock: t.Callable[[], float] = time.monotonic,
        dynamic_latency: bool = False,
    ) -> None:
        ...
        """
        params:
            * max_latency_in_ms: max_latency_in_ms for inbound tasks in milliseconds
            * max_batch_size: max batch size of inbound tasks
            * shared_sema: semaphore to limit concurrent outbound tasks
            * get_batch_size: callable to get batch size from inputs
            * batch_dim: tuple of (input_dim, output_dim) of batch
            * fallback: callable to return fallback result
            * dynamic_latency: whether requests may have different latency budgets
        raises:
            * all possible exceptions the decorated function has
        """
        self.max_latency = max_latency_in_ms / 1000.0
        self.fallback = fallback
        self.optimizer = Optimizer(self.max_latency)
        self.max_batch_size = int(max_batch_size)

        self._controller = None
        self._queue: collections.deque[Job[T_IN, T_OUT]] = (
            collections.deque()
        )  # TODO(bojiang): maxlen
        self._queue_revision = 0
        self._candidate_cache: tuple[int, CandidateInfo] | None = None
        self._dynamic_latency = dynamic_latency
        self._deadline_index_enabled = False
        self._deadline_heap: list[tuple[float, int]] = []
        self._active_job_tokens: set[int] = set()
        self._job_tokens: dict[int, int] = {}
        self._next_job_token = 0
        self._next_expiry: float | None = None
        self.get_batch_size = get_batch_size
        self.batch_dim = batch_dim
        self._clock = clock
        # at most 1 batch can be processed at the same time
        self._sema = shared_sema if shared_sema else NonBlockSema(1)

    def _invalidate_queue_state(self, *, recompute_expiry: bool) -> None:
        self._queue_revision += 1
        self._candidate_cache = None
        if recompute_expiry:
            self._rebuild_deadline_index()

    def _track_queued_job(self, job: Job[T_IN, T_OUT]) -> None:
        if not self._deadline_index_enabled:
            if self._next_expiry is None:
                self._next_expiry = job.deadline
            return
        token = self._next_job_token
        self._next_job_token += 1
        self._job_tokens[id(job)] = token
        self._active_job_tokens.add(token)
        heapq.heappush(self._deadline_heap, (job.deadline, token))
        if self._next_expiry is None or job.deadline < self._next_expiry:
            self._next_expiry = job.deadline

    def _untrack_queued_job(self, job: Job[T_IN, T_OUT]) -> None:
        if not self._deadline_index_enabled:
            return
        token = self._job_tokens.pop(id(job))
        self._active_job_tokens.remove(token)

    def _refresh_next_expiry(self) -> None:
        if not self._deadline_index_enabled:
            self._next_expiry = self._queue[0].deadline if self._queue else None
            return
        while (
            self._deadline_heap
            and self._deadline_heap[0][1] not in self._active_job_tokens
        ):
            heapq.heappop(self._deadline_heap)
        if len(self._deadline_heap) > 2 * len(self._active_job_tokens):
            self._rebuild_deadline_index()
            return
        self._next_expiry = self._deadline_heap[0][0] if self._deadline_heap else None

    def _rebuild_deadline_index(self) -> None:
        if not self._deadline_index_enabled:
            self._next_expiry = self._queue[0].deadline if self._queue else None
            return
        self._deadline_heap.clear()
        self._active_job_tokens.clear()
        self._job_tokens.clear()
        self._next_expiry = None
        for job in self._queue:
            self._track_queued_job(job)

    def _append_job(self, job: Job[T_IN, T_OUT]) -> None:
        if (
            not self._deadline_index_enabled
            and self._queue
            and job.deadline < self._queue[-1].deadline
        ):
            self._deadline_index_enabled = True
            self._rebuild_deadline_index()
        self._queue.append(job)
        self._invalidate_queue_state(recompute_expiry=False)
        self._track_queued_job(job)

    def _get_batch_size(self, data: T_IN) -> int:
        try:
            return self.get_batch_size(data)
        except Exception:
            logger.exception(
                "error in batch-size aware batching, falling back to regular batching method"
            )
            return 1

    def _peek_candidate(self, num_batches: int | None = None) -> CandidateInfo | None:
        use_cache = num_batches is None
        if use_cache and self._candidate_cache is not None:
            revision, candidate = self._candidate_cache
            if revision == self._queue_revision:
                return candidate

        limit = self.max_batch_size if num_batches is None else num_batches
        sample_size = 0
        job_count = 0
        first_enqueue_time = 0.0
        last_enqueue_time = 0.0
        earliest_deadline = float("inf")

        for job in self._queue:
            remaining = limit - sample_size
            if remaining <= 0:
                break
            batch_size = self._get_batch_size(job.data)
            if job_count == 0:
                first_enqueue_time = job.enqueue_time
            job_count += 1
            sample_size += min(batch_size, remaining)
            last_enqueue_time = job.enqueue_time
            earliest_deadline = min(earliest_deadline, job.deadline)
            if batch_size >= remaining:
                break

        if job_count == 0:
            return None

        candidate = CandidateInfo(
            job_count=job_count,
            sample_size=sample_size,
            first_enqueue_time=first_enqueue_time,
            last_enqueue_time=last_enqueue_time,
            earliest_deadline=earliest_deadline,
        )
        if use_cache:
            self._candidate_cache = (self._queue_revision, candidate)
        return candidate

    def _expire_jobs(self, now: float) -> int:
        if self._next_expiry is None or now < self._next_expiry:
            return 0

        if not self._deadline_index_enabled:
            expired: list[Job[T_IN, T_OUT]] = []
            while self._queue and self._queue[0].deadline <= now:
                expired.append(self._queue.popleft())
            self._invalidate_queue_state(recompute_expiry=True)
            for job in expired:
                if not job.future.done():
                    job.future.cancel()
            return len(expired)

        surviving: collections.deque[Job[T_IN, T_OUT]] = collections.deque()
        expired: list[Job[T_IN, T_OUT]] = []
        for job in self._queue:
            if job.deadline <= now:
                expired.append(job)
            else:
                surviving.append(job)

        self._queue = surviving
        self._invalidate_queue_state(recompute_expiry=True)
        for job in expired:
            if not job.future.done():
                job.future.cancel()
        return len(expired)

    def _take_inputs(
        self, num_batches: int | None = None
    ) -> tuple[Job[T_IN, T_OUT], ...]:
        inputs = tuple(self._get_inputs(num_batches))
        if inputs:
            self._invalidate_queue_state(recompute_expiry=False)
            self._refresh_next_expiry()
        return inputs

    def shutdown(self) -> None:
        if self._controller is not None:
            self._controller.cancel()
        for job in self._queue:
            job.future.cancel()

    @cached_property
    def _loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    @cached_property
    def _wake_event(self) -> asyncio.Condition:
        return asyncio.Condition()

    def __call__(
        self,
        callback: t.Callable[
            [t.Sequence[T_IN]], t.Coroutine[None, None, t.Sequence[T_OUT]]
        ],
    ) -> t.Callable[..., t.Coroutine[None, None, T_OUT | None]]:
        if is_async_callable(callback):
            self.callback = callback
        else:
            self.callback = functools.partial(run_in_threadpool, callback)

        @functools.wraps(callback)
        async def _func(data: T_IN, *, max_latency_ms: int | None = None) -> T_OUT:
            if self._controller is None:
                self._controller = self._loop.create_task(self.controller())
            try:
                r = await self.inbound_call(data, max_latency_ms=max_latency_ms)
            except asyncio.CancelledError:
                return self.fallback()
            if isinstance(r, Exception):
                raise r
            return r

        return _func

    async def train_optimizer(
        self,
        num_reqs_to_train: int,
        training_batch_size: int,
    ) -> None:
        target_job_count = min(self.max_batch_size, training_batch_size)
        req_count = 0
        try:
            while req_count < num_reqs_to_train:
                async with self._wake_event:
                    await self._wake_event.wait_for(self._queue.__len__)

                now = self._clock()
                self._expire_jobs(now)
                if not self._queue:
                    continue

                candidate = self._peek_candidate()
                assert candidate is not None
                a = self.optimizer.o_a
                b = self.optimizer.o_b
                predicted_duration = a * candidate.job_count + b
                oldest_wait = now - candidate.first_enqueue_time
                training_wait = (target_job_count * 2 + 1) * a + b
                adaptive_should_wait = (
                    candidate.job_count < target_job_count
                    and (target_job_count * a + b) + oldest_wait <= training_wait
                )
                decision = decide_next_action(
                    now=now,
                    candidate=candidate,
                    predicted_duration=predicted_duration,
                    adaptive_should_wait=adaptive_should_wait,
                    semaphore_locked=self._sema.is_locked(),
                    next_expiry=self._next_expiry,
                    ticket_interval=self.TICKET_INTERVAL,
                )
                if not decision.dispatch:
                    await asyncio.sleep(decision.wait_time)
                    continue

                acquired = self._sema.acquire()
                assert acquired
                inputs_info = self._take_inputs()
                req_count += 1
                self._loop.create_task(self.outbound_call(inputs_info, training=True))
        except Exception:
            logger.exception("Error in training optimizer")

    async def controller(self) -> None:
        """
        A standalone coroutine to wait/dispatch calling.
        """
        try:
            logger.debug("Starting dispatcher optimizer training...")
            # warm up the model
            await self.train_optimizer(self.optimizer.N_SKIPPED_SAMPLE + 6, 1)

            logger.debug("Dispatcher finished warming up model.")

            await self.train_optimizer(1, 1)
            self.optimizer.trigger_refresh()
            logger.debug("Dispatcher finished optimizer training request 1.")

            await self.train_optimizer(1, 2)
            self.optimizer.trigger_refresh()
            logger.debug("Dispatcher finished optimizer training request 2.")

            await self.train_optimizer(1, 3)
            self.optimizer.trigger_refresh()
            logger.debug("Dispatcher finished optimizer training request 3.")

            if self.optimizer.o_a + self.optimizer.o_b >= self.max_latency:
                logger.warning(
                    "BentoML has detected that a service has a max latency that is likely too low for serving. If many 503 errors are encountered, try raising the 'runner.max_latency' in your BentoML configuration YAML file."
                )
            logger.debug("Dispatcher optimizer training complete.")
        except Exception:  # pylint: disable=broad-except
            logger.exception("Error training optimizer")

        while True:
            try:
                async with self._wake_event:
                    await self._wake_event.wait_for(self._queue.__len__)

                now = self._clock()
                self._expire_jobs(now)
                if not self._queue:
                    continue

                if not self._dynamic_latency:
                    n = len(self._queue)
                    first_job = self._queue[0]
                    a = self.optimizer.o_a
                    b = self.optimizer.o_b
                    predicted_completion = now + a * n + b
                    must_dispatch = predicted_completion >= first_job.deadline
                    if self._sema.is_locked():
                        await asyncio.sleep(self.TICKET_INTERVAL)
                        continue
                    newest_wait = now - self._queue[-1].enqueue_time
                    if (
                        not must_dispatch
                        and n < self.max_batch_size
                        and n * (newest_wait + self.TICKET_INTERVAL + (a or 0))
                        <= self.optimizer.wait * 0.95
                    ):
                        await asyncio.sleep(self.TICKET_INTERVAL)
                        continue
                    acquired = self._sema.acquire()
                    assert acquired
                    inputs_info = self._take_inputs()
                    self._loop.create_task(self.outbound_call(inputs_info))
                    continue

                candidate = self._peek_candidate()
                assert candidate is not None
                a = self.optimizer.o_a
                b = self.optimizer.o_b
                predicted_duration = a * candidate.job_count + b
                newest_wait = now - candidate.last_enqueue_time
                adaptive_should_wait = (
                    candidate.sample_size < self.max_batch_size
                    and candidate.job_count
                    * (newest_wait + self.TICKET_INTERVAL + (a or 0))
                    <= self.optimizer.wait * 0.95
                )
                decision = decide_next_action(
                    now=now,
                    candidate=candidate,
                    predicted_duration=predicted_duration,
                    adaptive_should_wait=adaptive_should_wait,
                    semaphore_locked=self._sema.is_locked(),
                    next_expiry=self._next_expiry,
                    ticket_interval=self.TICKET_INTERVAL,
                )
                if not decision.dispatch:
                    await asyncio.sleep(decision.wait_time)
                    continue

                acquired = self._sema.acquire()
                assert acquired
                inputs_info = self._take_inputs()
                self._loop.create_task(self.outbound_call(inputs_info))
            except Exception:
                logger.exception("Error processing batch requests")

    async def inbound_call(
        self, data: T_IN, *, max_latency_ms: int | None = None
    ) -> T_OUT | Exception:
        now = self._clock()
        latency = (
            self.max_latency if max_latency_ms is None else max_latency_ms / 1000.0
        )
        if latency != self.max_latency:
            self._dynamic_latency = True
        future: asyncio.Future[T_OUT | Exception] = self._loop.create_future()
        input_info = Job(
            enqueue_time=now,
            data=data,
            future=future,
            deadline=now + latency,
        )
        self._append_job(input_info)
        async with self._wake_event:
            self._wake_event.notify_all()
        return await future

    async def outbound_call(
        self, inputs_info: tuple[Job[T_IN, T_OUT], ...], training: bool = False
    ):
        _time_start = self._clock()
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
                # fake wait as 0 for training requests
                wait=_time_start - inputs_info[0].enqueue_time if not training else 0,
                duration=self._clock() - _time_start,
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

    def _get_inputs(
        self, num_batches: int | None = None
    ) -> t.Iterable[Job[T_IN, T_OUT]]:
        if num_batches is None:
            num_batches = self.max_batch_size
        batch_size = 0
        while len(self._queue) > 0 and batch_size < num_batches:
            next_batch_size = self._get_batch_size(self._queue[0].data)
            if batch_size + next_batch_size <= num_batches:
                batch_size += next_batch_size
                job = self._queue.popleft()
                self._untrack_queued_job(job)
                yield job
            else:
                job = self._queue[0]
                chunk_size = num_batches - batch_size  # chunk_size < next_batch_size
                first_job, second_job = self.split_job(
                    job, [0, chunk_size, next_batch_size]
                )
                removed_job = self._queue.popleft()
                assert removed_job is job
                self._untrack_queued_job(job)
                batch_size += chunk_size
                self._queue.appendleft(second_job)
                self._track_queued_job(second_job)
                yield first_job

    def split_job(
        self, job: Job[T_IN, T_OUT], split_indexes: list[int]
    ) -> tuple[Job[T_IN, T_OUT], Job[T_IN, T_OUT]]:
        """Split the job into two child jobs and process them separately."""
        from ..runner.container import AutoContainer

        logger.debug(
            "Splitting batch into two child batches with indexes: %s", split_indexes
        )
        split_batches = AutoContainer.batch_to_batches(
            job.data, split_indexes, self.batch_dim[0]
        )
        assert len(split_batches) == 2
        first_child = attr.evolve(
            job, data=split_batches[0], future=self._loop.create_future()
        )
        second_child = attr.evolve(
            job, data=split_batches[1], future=self._loop.create_future()
        )

        def child_done_callback(fut: asyncio.Future[T_OUT | Exception]):
            if fut.cancelled():
                job.future.cancel()
                return
            if job.future.done():
                return
            result = fut.result()
            if isinstance(result, Exception):
                job.future.set_result(result)
                return
            if first_child.future.done() and second_child.future.done():
                result, _ = AutoContainer.batches_to_batch(
                    [first_child.future.result(), second_child.future.result()],
                    self.batch_dim[1],
                )
                job.future.set_result(result)

        first_child.future.add_done_callback(child_done_callback)
        second_child.future.add_done_callback(child_done_callback)
        return first_child, second_child
