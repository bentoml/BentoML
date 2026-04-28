"""Regression tests for bentoml.metrics deferred prometheus_client import.

Issue #5056: When bentoml.metrics is imported at service-module level (before
BentoML workers set PROMETHEUS_MULTIPROC_DIR), any Histogram/Counter objects
created afterwards were invisible to MultiProcessCollector because
prometheus_client had already initialised in single-process mode.

The fix defers the prometheus_client import inside __getattr__ so the import
always happens after the worker has set PROMETHEUS_MULTIPROC_DIR.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings


def _fresh_bentoml_metrics_module():
    """Return a fresh copy of the bentoml.metrics module, bypassing the cache.

    Necessary because bentoml.metrics caches imported attributes in globals()
    after the first access, and other tests may have already imported
    prometheus_client into the module namespace.
    """
    # Remove cached copies of bentoml.metrics and prometheus_client so we can
    # test the import-order behaviour in isolation.
    for key in list(sys.modules.keys()):
        if key == "bentoml.metrics" or key.startswith("prometheus_client"):
            del sys.modules[key]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import bentoml.metrics as m  # noqa: PLC0415

    return m


def test_bentoml_metrics_does_not_import_prometheus_client_eagerly():
    """Importing bentoml.metrics must NOT trigger prometheus_client import.

    This ensures the module can be imported at service-module level without
    locking prometheus_client into single-process mode before the worker has
    had a chance to set PROMETHEUS_MULTIPROC_DIR.
    """
    for key in list(sys.modules.keys()):
        if key == "bentoml.metrics" or key.startswith("prometheus_client"):
            del sys.modules[key]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import bentoml.metrics  # noqa: F401, PLC0415

    assert "prometheus_client" not in sys.modules, (
        "prometheus_client must not be imported at bentoml.metrics import time; "
        "it must be deferred until first attribute access so workers can set "
        "PROMETHEUS_MULTIPROC_DIR first"
    )


def test_multiple_histograms_all_collected_in_multiprocess_mode():
    """All user-defined Histogram objects must appear in MultiProcessCollector.

    Regression test for issue #5056: previously only the last declared
    Histogram was collected because prometheus_client was imported before
    PROMETHEUS_MULTIPROC_DIR was set, causing all metrics to be registered
    to the in-process registry instead of the file-backed multiprocess one.
    """
    m = _fresh_bentoml_metrics_module()

    # Simulate: worker sets PROMETHEUS_MULTIPROC_DIR AFTER the module is imported.
    tmp = tempfile.mkdtemp()
    old_env = os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    try:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = tmp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # First attribute access — prometheus_client is imported HERE,
            # after the env var is already set.
            Histogram = m.Histogram  # noqa: N806

        h1 = Histogram("test_latency_seconds", "Request latency")
        h2 = Histogram("test_image_width_pixels", "Image width")
        h3 = Histogram("test_image_height_pixels", "Image height")

        h1.observe(0.5)
        h2.observe(640.0)
        h3.observe(480.0)

        import prometheus_client  # noqa: PLC0415
        import prometheus_client.multiprocess  # noqa: PLC0415

        registry = prometheus_client.CollectorRegistry()
        prometheus_client.multiprocess.MultiProcessCollector(registry)
        collected_names = {m.name for m in registry.collect()}

        assert "test_latency_seconds" in collected_names, (
            "First histogram not found in MultiProcessCollector output (issue #5056)"
        )
        assert "test_image_width_pixels" in collected_names, (
            "Second histogram not found in MultiProcessCollector output (issue #5056)"
        )
        assert "test_image_height_pixels" in collected_names, (
            "Third histogram not found in MultiProcessCollector output (issue #5056)"
        )
    finally:
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        if old_env is not None:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = old_env
        # Clean up multiprocess DB files.
        import shutil  # noqa: PLC0415
        shutil.rmtree(tmp, ignore_errors=True)
        # Remove prometheus_client from sys.modules so other tests start clean.
        for key in list(sys.modules.keys()):
            if key.startswith("prometheus_client"):
                del sys.modules[key]
