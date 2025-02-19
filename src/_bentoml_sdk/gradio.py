from __future__ import annotations

import typing as t
from pathlib import Path

from bentoml.exceptions import MissingDependencyException

from .decorators import asgi_app

R = t.TypeVar("R")

try:
    import gradio as gr
    from gradio import Blocks
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """'gradio' is required by 'bentoml.gradio', run 'pip install -U gradio' to install gradio.""",
    )


def mount_gradio_app(blocks: Blocks, path: str, name: str = "gradio_ui"):
    """Mount a Gradio app to a BentoML service.

    Args:
        blocks: The Gradio blocks to be mounted.
        path: The URL path to mount the Gradio app.
        name: The name of the Gradio app.

    Example:

    Both of the following examples are allowed::

        @bentoml.gradio.mount_gradio_app(blocks)
        @bentoml.service()
        class MyService: ...

        @bentoml.service()
        @bentoml.gradio.mount_gradio_app(blocks)
        class MyService: ...

    """
    from _bentoml_sdk.service import Service
    from bentoml._internal import server

    favicon_path = str(
        Path(server.__file__).parent / "static_content" / "favicon-light-32x32.png"
    )
    assert path.startswith("/"), "Routed paths must start with '/'"
    path = path.rstrip("/")

    def decorator(obj: R) -> R:
        blocks.dev_mode = False
        blocks.show_error = True
        blocks.max_file_size = None
        blocks.validate_queue_settings()
        blocks.root_path = path
        blocks.favicon_path = favicon_path
        gradio_app = gr.routes.App.create_app(blocks, app_kwargs={"root_path": path})
        asgi_app(gradio_app, path=path, name=name)(obj)

        # @bentoml.service() decorator returns a wrapper instead of the original class
        # Check if the object is an instance of Service
        if isinstance(obj, Service):
            # For scenario:
            #
            # @bentoml.gradio.mount_gradio_app(..)
            # @bentoml.service(..)
            # class MyService: ...
            #
            # If the Service instance is already created, mount the ASGI app immediately
            target = obj.inner
        else:
            # For scenario:
            #
            # @bentoml.service(..)
            # @bentoml.gradio.mount_gradio_app(..)
            # class MyService: ...
            #
            # If the Service instance is not yet created, mark the Gradio app info
            # for later mounting during Service instance initialization
            target = obj

        # Store Gradio app information for ASGI app mounting and startup event callback
        gradio_apps = getattr(target, "__bentoml_gradio_apps__", [])
        gradio_apps.append((gradio_app, path, name))
        setattr(target, "__bentoml_gradio_apps__", gradio_apps)

        return obj

    return decorator
