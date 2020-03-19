import asyncio
from typing import Callable
from functools import lru_cache


class Parade:
    STATUSES = (STATUS_OPEN, STATUS_CLOSED, STATUS_RETURNED,) = range(3)

    def __init__(self, max_size, outbound_sema):
        self.max_size = max_size
        self.outbound_sema = outbound_sema
        self.batch_input = [None] * max_size
        self.batch_output = [None] * max_size
        self.cur = 0
        self.returned = asyncio.Condition()
        self.status = self.STATUS_OPEN

    def feed(self, data) -> Callable:
        '''
        feed data into this parade.
        return:
            the output index in parade.batch_output
        '''
        assert self.status == self.STATUS_OPEN
        self.batch_input[self.cur] = data
        self.cur += 1
        if self.cur == self.max_size:
            self.status = self.STATUS_CLOSED
        return self.cur - 1

    async def start_wait(self, interval, call):
        try:
            await asyncio.sleep(interval / 100)
            async with self.outbound_sema:
                self.status = self.STATUS_CLOSED
                self.batch_output = await call(self.batch_input[: self.cur])
                self.status = self.STATUS_RETURNED
                async with self.returned:
                    self.returned.notify_all()
        except Exception as e:  # noqa TODO
            raise e
        finally:
            # make sure parade is closed
            if self.status != self.STATUS_OPEN:
                self.status = self.STATUS_CLOSED


class ParadeDispatcher:
    def __init__(self, interval, max_size, task_concurrency=2):
        """
        params:
            * interval: interval to wait for inbound tasks in milliseconds
            * max_size: inbound tasks buffer size
            * task_concurrency: outbound tasks max concurrency
        """
        self.interval = interval
        self.max_size = max_size
        self.task_concurrency = task_concurrency
        self.callback = None
        self._current_parade = None

    @property
    @lru_cache(maxsize=1)
    def outbound_sema(self):
        '''
        semaphore should be created after forked(in the current event loop)
        '''
        return asyncio.Semaphore(self.task_concurrency)

    def get_parade(self):
        if self._current_parade and self._current_parade.status == Parade.STATUS_OPEN:
            return self._current_parade
        self._current_parade = Parade(self.max_size, self.outbound_sema)
        asyncio.get_event_loop().create_task(
            self._current_parade.start_wait(self.interval / 1000.0, self.callback)
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
