import sys
from pathlib import Path

import bentoml

THIS_DIR = Path(__file__).parent


# `workers: 1` is required, not a tuning choice: the command below binds a fixed
# port itself, so it cannot be replicated. Left unset, a custom-command service
# defaults to min(16, cpu/2) workers (see _bentoml_sdk/service/factory.py); only
# the first one starts the process, and the rest wait forever on a health check
# that can never pass if that one failed to bind — which hangs shutdown.
# The two services here also use different ports so that running them
# back-to-back cannot collide on a lingering child from the previous test.
StaticHTTP1 = bentoml.Service(
    "StaticHTTP1",
    cmd=[
        sys.executable,
        "-m",
        "http.server",
        "8000",
        "--directory",
        str(THIS_DIR),
    ],
    config={"endpoints": {"livez": "/"}, "workers": 1, "http": {"proxy_port": 8000}},
)


@bentoml.service(endpoints={"livez": "/"}, workers=1, http={"proxy_port": 8001})
class StaticHTTP2:
    def __command__(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "http.server",
            "8001",
            "--directory",
            str(THIS_DIR),
        ]

    def __metrics__(self, content: str) -> str:
        return f"{content}\n# HELLO from custom metrics\n"
