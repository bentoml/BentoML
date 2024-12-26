import os
import shutil
import socket
import sys
import sysconfig
from tempfile import mkdtemp
from tempfile import mkstemp

import tornado
from circus import get_arbiter
from circus.client import AsyncCircusClient
from circus.client import make_message
from circus.util import DEFAULT_ENDPOINT_DEALER
from circus.util import DEFAULT_ENDPOINT_SUB
from circus.util import ConflictError
from circus.util import tornado_sleep
from tornado.testing import AsyncTestCase
from zmq import ZMQError

DEBUG = sysconfig.get_config_var("Py_DEBUG") == 1

if "ASYNC_TEST_TIMEOUT" not in os.environ:
    os.environ["ASYNC_TEST_TIMEOUT"] = "30"


PYTHON = sys.executable

# Script used to sleep for a specified amount of seconds.
# Should be used instead of the 'sleep' command for
# compatibility
SLEEP = PYTHON + " -c 'import time;time.sleep(%d)'"


def get_ioloop():
    from tornado import ioloop

    return ioloop.IOLoop.current()


def get_available_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class TestCircus(AsyncTestCase):
    # how many times we will try when "Address already in use"
    ADDRESS_IN_USE_TRY_TIMES = 7

    arbiter_factory = get_arbiter
    arbiters = []

    def setUp(self):
        super(TestCircus, self).setUp()
        self.files = []
        self.dirs = []
        self.tmpfiles = []
        self._clients = {}
        self.plugins = []

    @property
    def cli(self):
        if self.arbiters == []:
            # nothing is running
            raise Exception("nothing is running")

        endpoint = self.arbiters[-1].endpoint
        if endpoint in self._clients:
            return self._clients[endpoint]

        cli = AsyncCircusClient(endpoint=endpoint)
        self._clients[endpoint] = cli
        return cli

    def _stop_clients(self):
        for client in self._clients.values():
            client.stop()
        self._clients.clear()

    def get_new_ioloop(self):
        return get_ioloop()

    def tearDown(self):
        for file in self.files + self.tmpfiles:
            try:
                os.remove(file)
            except OSError:
                pass
        for dir in self.dirs:
            try:
                shutil.rmtree(dir)
            except OSError:
                pass

        self._stop_clients()

        for plugin in self.plugins:
            plugin.stop()

        for arbiter in self.arbiters:
            if arbiter.running:
                try:
                    arbiter.stop()
                except ConflictError:
                    pass

        self.arbiters = []
        super(TestCircus, self).tearDown()

    def make_plugin(
        self,
        klass,
        endpoint=DEFAULT_ENDPOINT_DEALER,
        sub=DEFAULT_ENDPOINT_SUB,
        check_delay=1,
        **config,
    ):
        config["active"] = True
        plugin = klass(endpoint, sub, check_delay, None, **config)
        self.plugins.append(plugin)
        return plugin

    @tornado.gen.coroutine
    def start_arbiter(
        self, cmd="support.run_process", stdout_stream=None, debug=True, **kw
    ):
        testfile, arbiter = self._create_circus(
            cmd, stdout_stream=stdout_stream, debug=debug, use_async=True, **kw
        )
        self.test_file = testfile
        self.arbiter = arbiter
        self.arbiters.append(arbiter)
        for i in range(self.ADDRESS_IN_USE_TRY_TIMES):
            try:
                yield self.arbiter.start()
            except ZMQError as e:
                if e.strerror == "Address already in use":
                    # One more try to wait for the port being released by the OS
                    yield tornado_sleep(0.1)
                    continue
                else:
                    raise e
            else:
                # Everything goes well, just break
                break
        else:
            # Cannot start after tries
            raise RuntimeError(
                "Cannot start arbiter after %s times try"
                % self.ADDRESS_IN_USE_TRY_TIMES
            )

    @tornado.gen.coroutine
    def stop_arbiter(self):
        for watcher in self.arbiter.iter_watchers():
            yield self.arbiter.rm_watcher(watcher.name)
        yield self.arbiter._emergency_stop()

    @tornado.gen.coroutine
    def status(self, cmd, **props):
        resp = yield self.call(cmd, **props)
        raise tornado.gen.Return(resp.get("status"))

    @tornado.gen.coroutine
    def numwatchers(self, cmd, **props):
        resp = yield self.call(cmd, waiting=True, **props)
        raise tornado.gen.Return(resp.get("numprocesses"))

    @tornado.gen.coroutine
    def numprocesses(self, cmd, **props):
        resp = yield self.call(cmd, waiting=True, **props)
        raise tornado.gen.Return(resp.get("numprocesses"))

    @tornado.gen.coroutine
    def pids(self):
        resp = yield self.call("list", name="test")
        raise tornado.gen.Return(resp.get("pids"))

    def get_tmpdir(self):
        dir_ = mkdtemp()
        self.dirs.append(dir_)
        return dir_

    def get_tmpfile(self, content=None):
        fd, file = mkstemp()
        os.close(fd)
        self.tmpfiles.append(file)
        if content is not None:
            with open(file, "w") as f:
                f.write(content)
        return file

    @classmethod
    def _create_circus(
        cls,
        callable_path,
        plugins=None,
        stats=False,
        use_async=False,
        arbiter_kw=None,
        **kw,
    ):
        fd, testfile = mkstemp()
        os.close(fd)
        wdir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        args = ["circus/tests/generic.py", callable_path, testfile]
        worker = {
            "cmd": PYTHON,
            "args": args,
            "working_dir": wdir,
            "name": "test",
            "graceful_timeout": 2,
        }
        worker.update(kw)
        if not arbiter_kw:
            arbiter_kw = {}
        debug = arbiter_kw["debug"] = kw.get("debug", arbiter_kw.get("debug", False))
        # -1 => no periodic callback to manage_watchers by default
        arbiter_kw["check_delay"] = kw.get(
            "check_delay", arbiter_kw.get("check_delay", -1)
        )

        _gp = get_available_port
        arbiter_kw["controller"] = "tcp://127.0.0.1:%d" % _gp()
        arbiter_kw["pubsub_endpoint"] = "tcp://127.0.0.1:%d" % _gp()
        arbiter_kw["multicast_endpoint"] = "udp://237.219.251.97:12027"

        if stats:
            arbiter_kw["statsd"] = True
            arbiter_kw["stats_endpoint"] = "tcp://127.0.0.1:%d" % _gp()
            arbiter_kw["statsd_close_outputs"] = not debug

        if use_async:
            arbiter_kw["background"] = False
            arbiter_kw["loop"] = get_ioloop()
        else:
            arbiter_kw["background"] = True

        arbiter = cls.arbiter_factory([worker], plugins=plugins, **arbiter_kw)
        cls.arbiters.append(arbiter)
        return testfile, arbiter

    @tornado.gen.coroutine
    def _stop_runners(self):
        for arbiter in self.arbiters:
            yield arbiter.stop()
        self.arbiters = []

    @tornado.gen.coroutine
    def call(self, _cmd, **props):
        msg = make_message(_cmd, **props)
        resp = yield self.cli.call(msg)
        raise tornado.gen.Return(resp)
