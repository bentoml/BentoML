from __future__ import annotations

import contextlib
import logging
import typing as t

if t.TYPE_CHECKING:
    from _bentoml_sdk import Service
    from bentoml import Bento
    from bentoml import Tag


def load(
    bento_identifier: str | Tag | Bento,
    working_dir: str | None = None,
    reload: bool = False,
    standalone_load: bool = False,
) -> Service[t.Any]:
    """An alias for `bentoml.load` that ensures to return a new-style service object."""
    from bentoml._internal.service.loader import load as load_service

    return t.cast(
        "Service[t.Any]",
        load_service(
            bento_identifier,
            working_dir=working_dir,
            standalone_load=standalone_load,
            reload=reload,
        ),
    )


def import_service(
    svc_import_path: str,
    working_dir: str | None = None,
    reload: bool = False,
    standalone_load: bool = False,
) -> Service[t.Any]:
    """An alias for `bentoml._internal.services.loader.import_service` that ensures
    to return a new-style service object.
    """
    from bentoml._internal.service.loader import import_service as import_service_impl

    return t.cast(
        "Service[t.Any]",
        import_service_impl(
            svc_import_path,
            working_dir=working_dir,
            reload=reload,
            standalone_load=standalone_load,
        ),
    )


@contextlib.contextmanager
def importing() -> t.Iterator[None]:
    """A context manager that suppresses the ImportError when importing runtime-specific modules.

    Example:

    .. code-block:: python

        with bentoml.importing():
            import torch
            import tensorflow
    """
    from bentoml._internal.context import server_context

    logger = logging.getLogger("bentoml.service")

    try:
        yield
    except ImportError as e:
        if server_context.worker_index is not None:
            raise
        logger.info(
            "Skipping import of module %s because it is not available in the current environment",
            e.name,
        )
