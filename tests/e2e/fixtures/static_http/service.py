import sys
from pathlib import Path

import bentoml

THIS_DIR = Path(__file__).parent


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
    config={"endpoints": {"livez": "/"}},
)


@bentoml.service(endpoints={"livez": "/"})
class StaticHTTP2:
    def __command__(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "http.server",
            "8000",
            "--directory",
            str(THIS_DIR),
        ]

    def __metrics__(self, content: str) -> str:
        return f"{content}\n# HELLO from custom metrics\n"
