import attr


@attr.s
class _Runner:
    CPU: float = 1.0
    RAM: str = "100M"
    GPU: float = 0.0

    dynamic_batching = True
    max_batch_size = 10000
    max_latency_ms = 10000

    _on_gpu: bool = False

    def __init__(
        self,
        model_path: str,
        infer_api_callback: str,
        *,
        cpu: float = 1.0,
        ram: str = "100M",
        gpu: float = 0.0,
        dynamic_batching: bool = True,
        max_batch_size: int = 10000,
        max_latency_ms: int = 10000
    ):
        self._model_path = model_path
        self._infer_api_callback = infer_api_callback
        self.CPU = cpu
        self.RAM = ram
        self.GPU = gpu
        self.dynamic_batching = dynamic_batching
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
