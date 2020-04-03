import time
import aiohttp
import asyncio
import math
from collections import defaultdict


def dict_tab(d, in_row=False):
    try:
        from tabulate import tabulate

        if in_row:
            return tabulate(
                [(str(k), str(v)) for k, v in d.items()], tablefmt="fancy_grid"
            )
        else:
            return tabulate(
                (map(str, d.values()),),
                headers=map(str, d.keys()),
                tablefmt="fancy_grid",
            )
    except ImportError:
        return repr(d)


def percentile(data, p):
    if not data:
        return None
    size = len(data)
    return sorted(data)[max(math.ceil(size * p) - 1, 0)]


class Stat:
    def __init__(self):
        self.req_done = 0
        self.req_fail = 0
        self.req_times = []
        self._sess_start_time = 0
        self._sess_stop_time = 0
        self.client_busy = 0
        self.exceptions = defaultdict(lambda: 0)

    @property
    def req_time(self):
        return sum(self.req_times)

    @property
    def req_total(self):
        return self.req_fail + self.req_done

    @property
    def sess_time(self):
        if self._sess_stop_time:
            return self._sess_stop_time - self._sess_start_time
        else:
            return time.time() - self._sess_start_time

    def step(self):
        return {
            "Reqs": self.req_total,
            "Failure %": self.req_fail / max(self.req_total, 1) * 100,
            "Reqs/s": self.req_total / max(self.sess_time, 1),
            "Avg Resp Time": self.req_time / max(self.req_total, 1),
            "Client Health %": (1 - self.client_busy / max(self.req_total, 1)) * 100,
        }

    def sumup(self):
        r = {
            "Reqs": self.req_total,
            "Failure %": self.req_fail / max(self.req_total, 1) * 100,
            "Reqs/s": self.req_total / max(self.sess_time, 1),
            "Avg Resp Time": self.req_time / max(self.req_total, 1),
            "P50 Resp Time": percentile(self.req_times, 0.5),
            "P95": percentile(self.req_times, 0.95),
            "P99": percentile(self.req_times, 0.99),
            "Client Health %": (1 - self.client_busy / max(self.req_total, 1)) * 100,
        }
        if r["Client Health %"] < 90:
            print(
                f'''
                *** WARNNING ***
                The client health rate is low. The benchmark result is not reliable.
                Possible solutions:
                * check the failure_rate and avoid request failures
                * Rewrite your request_producer to reduce the CPU cost
                * Run more instances with multiprocessing. (Multi-threading will not
                                                            work because of the GIL)
                * Reduce the total_user of your session
                '''
            )

        return r


class BenchmarkClient:
    '''
    A locust-like benchmark tool with asyncio.
    Features:
    * Very effcient, low CPU cost
    * Could be embeded into other asyncio APPs, like jupyter notebook

    Paras:
    * request_producer: function with return value
        (url: str, method: str, headers: dict, data: str)
    * gen_wait_time: function to return each users' wait time between requests,
        for eg:
            - lambda: 1  # for constant 1 sec
            - lambda: random.random()  # for random wait time between 0 and 1
    * url_override: override the url provided by request_producer
    '''

    STATUS_STOPPED = 0
    STATUS_SPAWNING = 1
    STATUS_SPAWNED = 2
    STATUS_STOPPING = 3

    def __init__(self, request_producer, gen_wait_time, url_override=None, timeout=10):
        self.gen_wait_time = gen_wait_time
        self.request_producer = request_producer
        self.url_override = url_override
        self.user_pool = []
        self.status = self.STATUS_STOPPED
        self.stat = Stat()
        self.timeout = timeout
        self._stop_loop_flag = False

    async def _start(self):
        wait_time_suffix = 0
        url, method, headers, data = self.request_producer()
        async with aiohttp.ClientSession() as sess:
            while True:
                flag_collect_time = True
                req_start = time.time()
                req_url = self.url_override or url
                try:
                    async with sess.request(
                        method,
                        req_url,
                        data=data,
                        headers=headers,
                        timeout=self.timeout,
                    ) as r:
                        msg = await r.text()
                        if r.status // 100 == 2:
                            self.stat.req_done += 1
                        elif r.status // 100 == 4:
                            self.stat.exceptions[msg] += 1
                            self.stat.req_fail += 1
                        else:
                            flag_collect_time = False
                            self.stat.exceptions[msg] += 1
                            self.stat.req_fail += 1
                except asyncio.CancelledError:
                    raise
                except (
                    aiohttp.client_exceptions.ServerDisconnectedError,
                    TimeoutError,
                ) as e:
                    flag_collect_time = False
                    self.stat.exceptions[repr(e)] += 1
                    self.stat.req_fail += 1
                except Exception as e:  # pylint: disable=broad-except
                    flag_collect_time = False
                    self.stat.exceptions[repr(e)] += 1
                    self.stat.req_fail += 1

                req_stop = time.time()
                if flag_collect_time:
                    self.stat.req_times.append(req_stop - req_start)

                url, method, headers, data = self.request_producer()
                wait_time = self.gen_wait_time() + wait_time_suffix

                sleep_until = req_stop + wait_time

                await asyncio.sleep(sleep_until - time.time())

                now = time.time()
                if (now - sleep_until) / wait_time > 0.5:
                    self.stat.client_busy += 1
                wait_time_suffix = sleep_until - now

    def spawn(self):
        self.user_pool.append(asyncio.get_event_loop().create_task(self._start()))

    async def _trigger_batch_spawn(self, total, speed):
        try:
            wait_time = 1 / speed
            while self.status == self.STATUS_SPAWNING and len(self.user_pool) < total:
                self.spawn()
                await asyncio.sleep(wait_time)
        finally:
            if self.status == self.STATUS_SPAWNING:
                print(f"------ {total} users spawned ------")
                self.status = self.STATUS_SPAWNED
            else:
                print(f"------ spawn canceled before {total} users ------")

    def kill(self):
        if self.user_pool:
            self.user_pool.pop().cancel()
            if not self.user_pool:
                self.status = self.STATUS_STOPPED
            return True
        else:
            return False

    def batch_spawn(self, total, speed):
        if self.status in {self.STATUS_STOPPED}:
            self.status = self.STATUS_SPAWNING
            asyncio.get_event_loop().create_task(
                self._trigger_batch_spawn(total, speed)
            )
        else:
            print("The status must be STATUS_STOPPED before start_session/batch_spawn")

    def killall(self):
        if self.status in {self.STATUS_SPAWNING, self.STATUS_SPAWNED}:
            self.status = self.STATUS_STOPPING
        while self.kill():
            pass

    async def _start_output(self):
        while self.status in {self.STATUS_SPAWNING, self.STATUS_SPAWNED}:
            print('')
            print(dict_tab(self.stat.step()))
            await asyncio.sleep(2)

    async def _start_session(self, session_time, total_user, spawn_speed):
        try:
            print('======= Session started! =======')
            self.stat._sess_start_time = time.time()
            self.batch_spawn(total_user, spawn_speed)
            asyncio.get_event_loop().create_task(self._start_output())
            await asyncio.sleep(session_time)
        finally:
            self.killall()
            self.stat._sess_stop_time = time.time()
            print('======= Session stopped! =======')
            print(dict_tab(self.stat.sumup()))

            if self.stat.exceptions:
                print(f"------ Exceptions happened ------")
                print(dict_tab(self.stat.exceptions, in_row=True))

            if self._stop_loop_flag:
                loop = asyncio.get_event_loop()
                loop.stop()

    def start_session(self, session_time, total_user, spawn_speed):
        '''
        To start a benchmark session. If it's running It will return immediately.
        Paras:
        * session_time: session time in sec
        * total_user: user count need to be spawned
        * spawn_speed: user spawnning speed, in user/sec
        '''
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            self._stop_loop_flag = True
        loop.create_task(self._start_session(session_time, total_user, spawn_speed))
        if not loop.is_running():
            loop.run_forever()
