"""Deprecated metrics module — use prometheus_client directly.

This module is kept for backward compatibility. Importing from it still works,
but each attribute access defers the actual ``prometheus_client`` import to
call time rather than import time.

The deferred import is intentional: BentoML workers set
``PROMETHEUS_MULTIPROC_DIR`` in ``os.environ`` during startup, *after* the
service module is imported.  If ``prometheus_client`` were imported eagerly
here (at service-module import time), it would initialise in single-process
mode and never write to the multiprocess DB files — causing all user-defined
``Histogram`` / ``Counter`` objects to be invisible to
``MultiProcessCollector`` and the ``/metrics`` endpoint.

By deferring the import to first attribute access we ensure that
``prometheus_client`` always sees the correct ``PROMETHEUS_MULTIPROC_DIR``
value that the worker has already set.
"""

from __future__ import annotations

import warnings


def __getattr__(name: str) -> object:
    warnings.warn(
        "bentoml.metrics module is deprecated and will be removed in the future. "
        "Please use prometheus_client directly for metrics reporting.",
        DeprecationWarning,
        stacklevel=2,
    )
    import prometheus_client  # deferred — must happen after PROMETHEUS_MULTIPROC_DIR is set

    attr = getattr(prometheus_client, name)
    # Cache on this module so subsequent accesses skip __getattr__ entirely.
    globals()[name] = attr
    return attr
