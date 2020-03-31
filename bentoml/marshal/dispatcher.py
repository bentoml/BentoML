import asyncio
import time
import random
from typing import Callable
from bentoml.utils import cached_property
import numpy as np


class Bucket:
    '''
    Fixed size container.
    '''

    def __init__(self, size):
        self._data = [None] * size
        self._cur = 0
        self._size = size
        self._flag_full = False

    def put(self, v):
        self._data[self._cur] = v
        self._cur += 1
        if self._cur == self._size:
            self._cur = 0
            self._flag_full = True

    @property
    def data(self):
        if not self._flag_full:
            return self._data[: self._cur]
        return self._data

    def __len__(self):
        if not self._flag_full:
            return self._cur
        return self._size


class Optimizer:
    N_OUTBOUND_SAMPLE = 500
    N_OUTBOUND_WAIT_SAMPLE = 20

    def __init__(self):
        self.outbound_stat = Bucket(self.N_OUTBOUND_SAMPLE)
        self.outbound_wait_stat = Bucket(self.N_OUTBOUND_WAIT_SAMPLE)
        self.outbound_a = 0.0001
        self.outbound_b = 0
        self.outbound_wait = 0.01

    async def adaptive_wait(self, parade, max_time):
        dt = 0.001
        decay = 0.9
        while True:
            now = time.time()
            w0 = now - parade.time_first
            wn = now - parade.time_last
            n = parade.length
            a = max(self.outbound_a, 0)

            if w0 >= max_time:
                print("warning: max latency reached")
                break
            if max(n - 1, 1) * (wn + dt + a) <= self.outbound_wait * decay:
                await asyncio.sleep(dt)
                continue
            break

    def log_outbound_time(self, info):
        if info[0] < 5:  # skip all small batch
            return
        self.outbound_stat.put(info)
        if random.random() < 0.05:
            x = tuple((i, 1) for i, _ in self.outbound_stat.data)
            y = tuple(i for _, i in self.outbound_stat.data)
            self.outbound_a, self.outbound_b = np.linalg.lstsq(x, y, rcond=None)[0]

    def log_outbound_wait(self, info):
        self.outbound_wait_stat.put(info)
        self.outbound_wait = (
            sum(self.outbound_wait_stat.data) * 1.0 / len(self.outbound_wait_stat)
        )


class Parade:
    STATUSES = (STATUS_OPEN, STATUS_CLOSED, STATUS_RETURNED,) = range(3)

    def __init__(self, max_size, outbound_sema, optimizer):
        self.max_size = max_size
        self.outbound_sema = outbound_sema
        self.batch_input = [None] * max_size
        self.batch_output = [None] * max_size
        self.length = 0
        self.returned = asyncio.Condition()
        self.status = self.STATUS_OPEN
        self.optimizer = optimizer
        self.time_first = None
        self.time_last = None

    def feed(self, data) -> Callable:
        '''
        feed data into this parade.
        return:
            the output index in parade.batch_output
        '''
        self.batch_input[self.length] = data
        self.length += 1
        if self.length == self.max_size:
            self.status = self.STATUS_CLOSED
        self.time_last = time.time()
        return self.length - 1

    async def start_wait(self, max_wait_time, call):
        now = time.time()
        self.time_first = now
        self.time_last = now
        try:
            await self.optimizer.adaptive_wait(self, max_wait_time)
            async with self.outbound_sema:
                self.status = self.STATUS_CLOSED
                _time_start = time.time()
                self.optimizer.log_outbound_wait(_time_start - self.time_first)
                self.batch_output = await call(self.batch_input[: self.length])
                self.optimizer.log_outbound_time(
                    (self.length, time.time() - _time_start)
                )
                self.status = self.STATUS_RETURNED
        finally:
            # make sure parade is closed
            if self.status == self.STATUS_OPEN:
                self.status = self.STATUS_CLOSED
            async with self.returned:
                self.returned.notify_all()


class ParadeDispatcher:
    def __init__(self, max_wait_time, max_size, shared_sema: callable = None):
        """
        params:
            * max_wait_time: max_wait_time to wait for inbound tasks in milliseconds
            * max_size: inbound tasks buffer size
            * shared_sema: semaphore to limit concurrent tasks
        """
        self.max_wait_time = max_wait_time
        self.max_size = max_size
        self.shared_sema = shared_sema
        self.callback = None
        self._current_parade = None
        self.optimizer = Optimizer()

    @cached_property
    def outbound_sema(self):
        '''
        semaphore should be created after process forked
        '''
        return self.shared_sema() if self.shared_sema else asyncio.Semaphore(1)

    def get_parade(self):
        if self._current_parade and self._current_parade.status == Parade.STATUS_OPEN:
            return self._current_parade
        self._current_parade = Parade(self.max_size, self.outbound_sema, self.optimizer)
        asyncio.get_event_loop().create_task(
            self._current_parade.start_wait(self.max_wait_time / 1000.0, self.callback)
        )
        return self._current_parade

    def __call__(self, callback):
        self.callback = callback

        async def _func(inputs):
            parade = self.get_parade()
            _id = parade.feed(inputs)
            async with parade.returned:
                await parade.returned.wait()
            return parade.batch_output[_id]

        return _func
