import asyncio
import math
import time
from collections import defaultdict

from tabulate import tabulate


def wrap_line(s, line_width=120, sep='\n'):
    ls = s.split(sep)
    outs = []
    for line in ls:
        while len(line) > line_width:
            outs.append(line[:line_width])
            line = line[line_width:]
        outs.append(line)
    return sep.join(outs)


def dict_tab(d, in_row=False):
    try:
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


def percentile(data, pers):
    if not data:
        return [math.nan] * len(pers)
    size = len(data)
    sorted_data = sorted(data)
    return tuple(sorted_data[max(math.ceil(size * p) - 1, 0)] for p in pers)


class DynamicBucketMerge:
    '''
    real time speed stat
    '''

    def __init__(self, sample_range=1, bucket_num=10):
        self.bucket_num = bucket_num
        self.sample_range = sample_range
        self.bucket = [0] * bucket_num
        self.bucket_sample = [0] * bucket_num
        self.bucket_ver = [0] * bucket_num

    def put(self, timestamp, num):
        timestamp = timestamp / self.sample_range
        ver = int(timestamp * self.bucket_num // 1)
        i = int(timestamp % 1 * self.bucket_num // 1)
        if ver > self.bucket_ver[i]:
            self.bucket_ver[i], self.bucket[i], self.bucket_sample[i] = ver, num, 1
        else:
            self.bucket[i] += num
            self.bucket_sample[i] += 1

    def sum(self, timestamp):
        timestamp = timestamp / self.sample_range
        ver = int(timestamp * self.bucket_num // 1)
        return (
            sum(
                n
                for n, v in zip(self.bucket, self.bucket_ver)
                if v >= ver - self.bucket_num
            )
            / self.sample_range
        )

    def mean(self, timestamp):
        timestamp = timestamp / self.sample_range
        ver = int(timestamp * self.bucket_num // 1)
        num_n_count = [
            (n, s)
            for n, s, v in zip(self.bucket, self.bucket_sample, self.bucket_ver)
            if v >= ver - self.bucket_num and s > 0
        ]
        return (
            sum(n for n, _ in num_n_count) / sum(s for _, s in num_n_count)
            if num_n_count
            else math.nan
        )


class Stat:
    def __init__(self):
        self.success = 0
        self.fail = 0
        self.succ_ps = DynamicBucketMerge(2, 10)
        self.exec_ps = DynamicBucketMerge(2, 10)
        self.succ_times = []
        self.exec_times = []
        self.succ_time_ps = DynamicBucketMerge(2, 10)
        self.exec_time_ps = DynamicBucketMerge(2, 10)
        self.client_busy = 0
        self.exceptions = defaultdict(list)
        self._sess_start_time = 0
        self._sess_stop_time = 0

    @property
    def req_total(self):
        return self.fail + self.success

    @property
    def sess_time(self):
        if self._sess_stop_time:
            return self._sess_stop_time - self._sess_start_time
        else:
            return time.time() - self._sess_start_time

    def log_succeed(self, req_time, n=1):
        self.success += n
        self.succ_ps.put(time.time(), 1)
        self.succ_times.append(req_time)
        self.succ_time_ps.put(time.time(), req_time)

    def log_exception(self, group, msg, req_time, n=1):
        self.fail += n
        self.exec_ps.put(time.time(), 1)
        self.exec_times.append(req_time)
        self.exec_time_ps.put(time.time(), req_time)
        self.exceptions[group].append(msg)

    def print_step(self):
        now = time.time()
        headers = (
            "Result",
            "Total",
            "Reqs/s",
            "Resp Time Avg",
            "Client Health %",
        )
        r = (
            (
                "succ",
                f"{self.success}",
                f"{self.succ_ps.sum(now)}",
                f"{self.succ_time_ps.mean(now)}",
                f"{(1 - self.client_busy / max(self.req_total, 1)) * 100}",
            ),
            (
                "fail",
                f"{self.fail}",
                f"{self.exec_ps.sum(now)}",
                f"{self.exec_time_ps.mean(now)}",
                "",
            ),
        )

        print(tabulate(r, headers=headers, tablefmt="fancy_grid"))

    def print_sumup(self):
        ps = percentile(self.succ_times, [0.5, 0.95, 0.99])
        ps_fail = percentile(self.exec_times, [0.5, 0.95, 0.99])
        health = (1 - self.client_busy / max(self.req_total, 1)) * 100
        headers = (
            "Result",
            "Total",
            "Reqs/s",
            "Resp Time Avg",
            "P50",
            "P95",
            "P99",
        )
        r = (
            (
                "succ",
                self.success,
                self.success / max(self.sess_time, 1),
                sum(self.succ_times) / max(self.success, 1),
                ps[0],
                ps[1],
                ps[2],
            ),
            (
                "fail",
                self.fail,
                self.fail / max(self.sess_time, 1),
                sum(self.exec_times) / max(self.fail, 1),
                ps_fail[0],
                ps_fail[1],
                ps_fail[2],
            ),
        )

        print(tabulate(r, headers=headers, tablefmt="fancy_grid"))

        print(f"------ Client Health {health:.1f}% ------")
        if health < 90:
            print(
                """
                *** WARNING ***
                The client health rate is low. The benchmark result is not reliable.
                Possible solutions:
                * check the failure_rate and avoid request failures
                * Rewrite your request_producer to reduce the CPU cost
                * Run more instances with multiprocessing. (Multi-threading will not
                                                            work because of the GIL)
                * Reduce the total_user of your session
                """
            )

    def print_exec(self):
        headers = ['exceptions', 'count']
        rs = [
            (wrap_line(str(v[0]), 50)[:1000], len(v),)
            for k, v in self.exceptions.items()
        ]
        print(tabulate(rs, headers=headers, tablefmt='fancy_grid'))


def default_verify_response(status, _):
    if status // 100 == 2:
        return True
    else:
        return False


class BenchmarkClient:
    """
    A locust-like benchmark tool with asyncio.
    Features:
    * Very efficient, low CPU cost
    * Could be embedded into other asyncio apps, like jupyter notebook

    Paras:
    * request_producer: The test case producer, a function with return value
        (url: str, method: str, headers: dict, data: str)
    * request_interval: intervals in seconds between each requests of the same user,
      lazy value supported.
        for eg:
            - 1  # for constant 1 sec
            - lambda: random.random()  # for random wait time between 0 and 1
    * url_override: override the url provided by request_producer


    Example usage
    =========

    In a session of one minute, 100 users keep sending POST request with
    one seconds interval:

    ``` test.py
    def test_case_producer():
        return ('http://localhost:5000',
                "POST",
                {"Content-Type": "application/json"},
                '{"x": 1.0}')

    from bentoml.utils.benchmark import BenchmarkClient
    b = BenchmarkClient(test_case_producer, request_interval=1, timeout=10)
    b.start_session(session_time=60, total_user=100)
    ```

    run command:
    > python test.py

    """

    STATUS_STOPPED = 0
    STATUS_SPAWNING = 1
    STATUS_SPAWNED = 2
    STATUS_STOPPING = 3

    def __init__(
        self,
        request_producer: callable,
        request_interval,
        verify_response: callable = default_verify_response,
        url_override=None,
        timeout=10,
    ):
        self.request_interval = request_interval
        self.request_producer = request_producer
        self.verify_response = verify_response
        self.url_override = url_override
        self.user_pool = []
        self.status = self.STATUS_STOPPED
        self.stat = Stat()
        self.timeout = timeout
        self._stop_loop_flag = False

    async def _start(self):
        from aiohttp import ClientSession
        from aiohttp.client_exceptions import ServerDisconnectedError

        wait_time_suffix = 0
        url, method, headers, data = self.request_producer()
        async with ClientSession() as sess:
            while True:
                req_start = time.time()
                req_url = self.url_override or url
                err = ''
                group = ''
                # noinspection PyUnresolvedReferences
                try:
                    async with sess.request(
                        method,
                        req_url,
                        data=data,
                        headers=headers,
                        timeout=self.timeout,
                    ) as r:
                        msg = await r.text()
                        if not self.verify_response(r.status, msg):
                            group = f"{r.status}"
                            err = f"<status: {r.status}>\n{msg}"
                except asyncio.CancelledError:  # pylint: disable=try-except-raise
                    raise
                except (ServerDisconnectedError, TimeoutError) as e:
                    group = repr(e.__class__)
                    err = repr(e)
                except Exception as e:  # pylint: disable=broad-except
                    group = repr(e.__class__)
                    err = repr(e)

                req_stop = time.time()
                if err:
                    self.stat.log_exception(group, err, req_stop - req_start)
                else:
                    self.stat.log_succeed(req_stop - req_start)

                url, method, headers, data = self.request_producer()
                if callable(self.request_interval):
                    request_interval = self.request_interval()
                else:
                    request_interval = self.request_interval
                wait_time = request_interval + wait_time_suffix

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
            print("")
            self.stat.print_step()
            await asyncio.sleep(2)

    async def _start_session(self, session_time, total_user, spawn_speed):
        try:
            print("======= Session started! =======")
            self.stat._sess_start_time = time.time()
            self.batch_spawn(total_user, spawn_speed)
            asyncio.get_event_loop().create_task(self._start_output())
            await asyncio.sleep(session_time)
        finally:
            self.killall()
            self.stat._sess_stop_time = time.time()
            print("======= Session stopped! =======")
            if self.stat.exceptions:
                print("------ Exceptions happened ------")
                self.stat.print_exec()

            self.stat.print_sumup()

            if self._stop_loop_flag:
                loop = asyncio.get_event_loop()
                loop.stop()

    def start_session(self, session_time, total_user, spawn_speed=None):
        """
        To start a benchmark session. If it's running It will return immediately.
        Paras:
        * session_time: session time in sec
        * total_user: user count need to be spawned
        * spawn_speed: user spawning speed, in user/sec
        """
        if spawn_speed is None:
            spawn_speed = total_user
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            self._stop_loop_flag = True
        loop.create_task(self._start_session(session_time, total_user, spawn_speed))
        if not loop.is_running():
            loop.run_forever()
