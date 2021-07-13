import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NoopTracer:
    def __init__(self):
        logger.debug("Tracing is disabled. Initializing no-op tracer")

    @contextmanager
    def span(self, *args, **kwargs):  # pylint: disable=unused-argument
        yield
        return

    @contextmanager
    def async_span(self, *args, **kwargs):  # pylint: disable=unused-argument
        yield
        return
