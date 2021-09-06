import attr


@attr.s
class Runner:
    _model_path = attr.ib()
    _infer_api_callback = attr.ib()

    CPU = attr.ib(default=1.0, type=float)
    RAM = attr.ib(default="100M", type=str)

    GPU = attr.ib(default=0.0, type=float)
    _on_gpu = attr.ib(default=False, type=bool)

    dynamic_batching = attr.ib(default=True, type=bool)
    max_batch_size = attr.ib(default=10000, type=int)
    max_latency_ms = attr.ib(default=10000, type=int)
