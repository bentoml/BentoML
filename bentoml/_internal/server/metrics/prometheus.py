import os
import shutil
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
        this function is called. It relies on the os.environ['prometheus_multiproc_dir']
        to properly setup for multiprocess mode
        """
        self.multiproc = multiproc
        self.namespace = namespace
        self._registry = None

        if multiproc:
            assert multiproc_dir is not None, "multiproc_dir must be provided"
            os.environ["prometheus_multiproc_dir"] = multiproc_dir

    @classmethod
    def clean_multiproc_dir(cls, multiproc_dir: t.Optional[str]):
        assert multiproc_dir is not None, "multiproc_dir must be provided"
        logger.debug("Setting up prometheus_multiproc_dir: %s", multiproc_dir)
        # Wipe prometheus metrics directory between runs
        # https://github.com/prometheus/client_python#multiprocess-mode-gunicorn
        # Ignore errors so it does not fail when directory does not exist
        shutil.rmtree(multiproc_dir, ignore_errors=True)
        os.makedirs(multiproc_dir, exist_ok=True)

    @property
    def registry(self):
        if self._registry is None:
            from prometheus_client import multiprocess, CollectorRegistry

            registry = CollectorRegistry()
            if self.multiproc:
                multiprocess.MultiProcessCollector(registry)
            self._registry = registry
        return self._registry

    # review
    # @classmethod
    # def mark_process_dead(cls, pid: int):
    #     # TODO(jiang)
    #     pass

    @staticmethod
    def mark_process_dead(pid: int) -> None:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(pid)

    def start_http_server(self, port: int, addr: str = "") -> None:
        from prometheus_client import start_http_server

        start_http_server(port=port, addr=addr, registry=self.registry)

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
