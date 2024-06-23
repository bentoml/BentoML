import sys
import warnings

import prometheus_client

warnings.warn(
    "bentoml.metrics module is deprecated and will be removed in the future. "
    "Please use prometheus_client directly for metrics reporting.",
    DeprecationWarning,
    stacklevel=1,
)

sys.modules[__name__] = prometheus_client
