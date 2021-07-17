import multiprocessing
from bentoml.configuration.containers import BentoMLContainer as BentoMLContainer
from typing import Any

Lock = multiprocessing.synchronize.Lock
logger: Any

class InstrumentMiddleware:
    app: Any
    bento_service: Any
    metrics_request_duration: Any
    metrics_request_total: Any
    metrics_request_in_progress: Any
    def __init__(self, app, bento_service, metrics_client=...) -> None: ...
    def __call__(self, environ, start_response): ...
