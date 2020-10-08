import asyncio
import time
import random
from bentoml.marshal.dispatcher import CorkDispatcher


class Session:
    def __init__(self, config):
        self.config = config
        self._req_sucess = 0
        self._req_cancel = 0
        self._req_err = 0
        self._req_not_finished = 0
        self._req_timeout = 0
        self._time_start = 0
        self._stop_flag = False
        self._result_fut = asyncio.get_event_loop().create_future()
        self._users = []

        @CorkDispatcher(config.MAX_EXPECTED_TIME, config.MAX_BATCH_SIZE)
        async def _work(xs):
            x = len(xs)
            await asyncio.sleep(config.A * x + config.B)
            return xs

        self.work = _work

    async def user_life(self):
        try:
            while time.time() - self._time_start <= self.config.time_total:
                now = time.time()
                i = random.randint(1, 1000)
                r = await self.work(i)
                if r == i and time.time() - now < self.config.timeout:
                    self._req_sucess += 1
                elif r == i:
                    self._req_timeout += 1
                elif r is None:
                    self._req_cancel += 1
                else:
                    self._req_err += 1
                await asyncio.sleep(self.config.interval)
        except asyncio.CancelledError:
            self._req_not_finished += 1

    async def clock(self):
        self._time_start = time.time()
        await asyncio.sleep(self.config.time_total)
        self._stop_flag = True
        for user in self._users:
            user.cancel()
        await asyncio.sleep(0)
        _time_used = time.time() - self._time_start
        self._result_fut.set_result(
            dict(
                req_sucess=self._req_sucess,
                req_timeoout=self._req_timeout,
                req_cancel=self._req_cancel,
                req_err=self._req_err,
                req_not_finished=self._req_not_finished,
                time_total=_time_used,
            )
        )

    async def __call__(self):
        asyncio.get_event_loop().create_task(self.clock())
        for _ in range(self.config.user_total):
            if self._stop_flag:
                break
            self._users.append(asyncio.get_event_loop().create_task(self.user_life()))
            await asyncio.sleep(self.config.spawn_interval)
        r = await self._result_fut
        assert r['req_err'] == 0
        print(r)


def async_stop(loop):
    for task in asyncio.Task.all_tasks():
        task.cancel()
    loop.run_until_complete(asyncio.sleep(0))


class Config1:
    '''
    low presure
    '''

    time_total = 20
    user_total = 10
    timeout = 10

    spawn_interval = 10 / user_total
    interval = 1
    A = 0.002
    B = 0.1
    MAX_BATCH_SIZE = 10000
    MAX_EXPECTED_TIME = timeout * 1000 * 0.95


class Config2:
    '''
    medium presure
    '''

    time_total = 20
    user_total = 400
    timeout = 10

    spawn_interval = 10 / user_total
    interval = 1
    A = 0.002
    B = 0.0
    MAX_BATCH_SIZE = 10000
    MAX_EXPECTED_TIME = timeout * 1000 * 0.95


class Config3:
    '''
    high presure
    '''

    time_total = 20
    user_total = 2000
    timeout = 10

    spawn_interval = 10 / user_total
    interval = 1
    A = 0.002
    B = 0.1
    MAX_BATCH_SIZE = 10000
    MAX_EXPECTED_TIME = timeout * 1000 * 0.95


class Config4:
    '''
    slow outbound request
    '''

    time_total = 20
    user_total = 1000
    timeout = 10

    spawn_interval = 10 / user_total
    interval = 1
    A = 1
    B = 0.1
    MAX_BATCH_SIZE = 10000
    MAX_EXPECTED_TIME = timeout * 1000 * 0.95


class Config5:
    '''
    dropping performance
    '''

    time_total = 20
    user_total = 10000
    timeout = 10

    spawn_interval = 10 / user_total
    interval = 1
    A = 30
    B = 0.1
    MAX_BATCH_SIZE = 10000
    MAX_EXPECTED_TIME = timeout * 1000 * 0.95


def main():
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(Session(Config5)())
        async_stop(loop)
        loop.run_until_complete(Session(Config4)())
        async_stop(loop)
        loop.run_until_complete(Session(Config3)())
        async_stop(loop)
        loop.run_until_complete(Session(Config2)())
        async_stop(loop)
        loop.run_until_complete(Session(Config1)())
        async_stop(loop)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
