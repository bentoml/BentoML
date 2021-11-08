from contextlib import contextmanager


class Tracer:
    def __init__(self) -> None:
        pass

    @contextmanager
    def span(self, *args, **kwargs):  # pylint: disable=unused-argument
        yield

    @contextmanager
    def async_span(self, *args, **kwargs):  # pylint: disable=unused-argument
        yield
