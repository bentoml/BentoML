from ._internal.cloud import BentoCloudClient as BentoCloudClient
from ._internal.cloud import YataiClient as YataiClient

deprecated_names = ["Resource"]


def __getattr__(name: str):
    if name in deprecated_names:
        raise AttributeError(
            f"{name} is deprecated, please use bentoml.deloyment instead"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
