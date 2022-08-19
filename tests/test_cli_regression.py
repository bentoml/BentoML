from __future__ import annotations

import time
import subprocess


def test_regression():
    # note that this should only be run in a single process.
    start = time.perf_counter()
    subprocess.run(["bentoml"])
    finish = time.perf_counter() - start

    # The actually CLI runtime are around 368ms for
    # just running the CLI entrypoint.
    # The upper value is a loose margin that can be accepted
    # factoring in subprocess overhead.
    assert finish <= 0.43
