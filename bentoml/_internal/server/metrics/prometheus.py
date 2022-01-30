# type: ignore[reportMissingTypeStubs]
import os
import sys
import typing as t
import logging
from functools import partial

logger = logging.getLogger(__name__)


class PrometheusClient:
    def __init__(
        self,
        *,
        namespace: str = "",
        multiproc: bool = True,
        multiproc_dir: t.Optional[str] = None,
    ):
        """
        Set up multiproc_dir for prometheus to work in multiprocess mode,
        which is required when working with Gunicorn server

        Warning: for this to work, prometheus_client library must be imported after
        this function is called. It relies on the os.environ['PROMETHEUS_MULTIPROC_DIR']
        to properly setup for multiprocess mode
        """
        if multiproc:
            assert multiproc_dir is not None, "multiproc_dir must be provided"

        self.multiproc = multiproc
        self.namespace = namespace
        self.multiproc_dir: t.Optional[str] = None
        self._registry = None
        self._pid: t.Optional[int] = None

    @property
    def registry(self):
        if self._registry is None:
            if self.multiproc:
                # step 1: check environment
                assert (
                    "prometheus_client" not in sys.modules
                ), "prometheus_client is already imported, multiprocessing will not work properly"

                assert self.multiproc_dir
                assert os.path.isdir(self.multiproc_dir)

                os.environ["PROMETHEUS_MULTIPROC_DIR"] = self.multiproc_dir

                # step 2:
                from prometheus_client import multiprocess
                from prometheus_client import CollectorRegistry

                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                self._pid = os.getpid()
                self._registry = registry
            else:
                from prometheus_client import REGISTRY as registry

                self._registry = registry
        else:
            if self.multiproc:
                assert self._pid is not None
                assert (
                    os.getpid() == self._pid
                ), "The current process's different than the process which the prometheus client gets created"

        return self._registry

    def mark_process_dead(self) -> None:
        if self.multiproc:
            assert self._pid is not None
            assert (
                os.getpid() == self._pid
            ), "The current process's different than the process which the prometheus client gets created"
            from prometheus_client import multiprocess

            multiprocess.mark_process_dead(self._pid)

    # def start_http_server(self, port: int, addr: str = "") -> None:
    # from prometheus_client import start_http_server

    # start_http_server(port=port, addr=addr, registry=self.registry)

    def generate_latest(self):
        from prometheus_client import generate_latest

        return generate_latest(self.registry)

    @property
    def CONTENT_TYPE_LATEST(self) -> str:
        from prometheus_client import CONTENT_TYPE_LATEST

        return CONTENT_TYPE_LATEST

    @property
    def Histogram(self):
        from prometheus_client import Histogram as Operator

        return partial(Operator, namespace=self.namespace, registry=self.registry)

    @property
    def Counter(self):
        from prometheus_client import Counter as Operator

        return partial(Operator, namespace=self.namespace, registry=self.registry)

    @property
    def Summary(self):
        from prometheus_client import Summary as Operator

        return partial(Operator, namespace=self.namespace, registry=self.registry)

    @property
    def Gauge(self):
        from prometheus_client import Gauge as Operator

        return partial(Operator, namespace=self.namespace, registry=self.registry)
