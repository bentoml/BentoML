"""
BentoML
=======

BentoML is a Python library for building online serving systems optimized for AI apps
and model inference. It supports serving any model format/runtime and custom Python
code, offering the key primitives for serving optimizations, task queues, batching,
multi-model chains, distributed orchestration, and multi-GPU serving.

Docs: http://docs.bentoml.com
Source Code: https://github.com/bentoml/BentoML
Developer Community: https://l.bentoml.com/join-slack
Twitter/X: https://x.com/bentomlai
Blog: https://www.bentoml.com/blog
"""

from typing import TYPE_CHECKING
from typing import Any

from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_config
from ._internal.configuration import save_config
from ._internal.configuration import set_serialization_strategy

# Inject dependencies and configurations
load_config()

from pydantic import Field

# BentoML built-in types
from ._internal.bento import Bento
from ._internal.cloud import BentoCloudClient
from ._internal.context import ServiceContext as Context
from ._internal.context import server_context
from ._internal.models import Model
from ._internal.monitoring import monitor
from ._internal.resource import Resource
from ._internal.runner import Runnable
from ._internal.runner import Runner
from ._internal.runner.strategy import Strategy
from ._internal.service import Service
from ._internal.service.loader import load
from ._internal.tag import Tag
from ._internal.utils.http import Cookie

# Bento management APIs
from .bentos import build
from .bentos import delete
from .bentos import export_bento
from .bentos import get
from .bentos import import_bento
from .bentos import list
from .bentos import pull
from .bentos import push
from .bentos import serve

if TYPE_CHECKING:
    # Framework specific modules
    from _bentoml_impl.frameworks import catboost
    from _bentoml_impl.frameworks import lightgbm
    from _bentoml_impl.frameworks import mlflow
    from _bentoml_impl.frameworks import sklearn
    from _bentoml_impl.frameworks import xgboost

    try:  # needs bentoml-unsloth package
        from _bentoml_impl.frameworks import unsloth
    except ModuleNotFoundError:
        pass

    from . import diffusers_simple
    from . import gradio
    from . import ray
    from . import triton
    from ._internal.frameworks import detectron
    from ._internal.frameworks import diffusers
    from ._internal.frameworks import easyocr
    from ._internal.frameworks import fastai
    from ._internal.frameworks import flax
    from ._internal.frameworks import keras
    from ._internal.frameworks import onnx
    from ._internal.frameworks import picklable_model
    from ._internal.frameworks import pytorch
    from ._internal.frameworks import pytorch_lightning
    from ._internal.frameworks import tensorflow
    from ._internal.frameworks import torchscript
    from ._internal.frameworks import transformers

    # isort: off
    from . import io
    from . import models
    from . import metrics  # Prometheus metrics client
    from . import container  # Container API
    from . import client  # Client API
    from . import batch  # Batch API
    from . import exceptions  # BentoML exceptions
    from . import monitoring  # Monitoring API
    from . import cloud  # Cloud API
    from . import deployment  # deployment API
    from . import validators  # validators

    # isort: on
    from _bentoml_impl.client import AsyncHTTPClient
    from _bentoml_impl.client import SyncHTTPClient
    from _bentoml_impl.loader import importing
    from _bentoml_sdk import IODescriptor
    from _bentoml_sdk import api
    from _bentoml_sdk import asgi_app
    from _bentoml_sdk import depends
    from _bentoml_sdk import get_current_service
    from _bentoml_sdk import images
    from _bentoml_sdk import mount_asgi_app
    from _bentoml_sdk import on_deployment
    from _bentoml_sdk import on_shutdown
    from _bentoml_sdk import runner_service
    from _bentoml_sdk import service
    from _bentoml_sdk import task
else:
    from _bentoml_impl.frameworks.importer import FrameworkImporter

    from ._internal.utils import LazyLoader as _LazyLoader
    from ._internal.utils.pkg import pkg_version_info

    FrameworkImporter.install()

    # ML Frameworks
    catboost = _LazyLoader(
        "bentoml.catboost", globals(), "_bentoml_impl.frameworks.catboost"
    )
    sklearn = _LazyLoader(
        "bentoml.sklearn", globals(), "_bentoml_impl.frameworks.sklearn"
    )
    xgboost = _LazyLoader(
        "bentoml.xgboost", globals(), "_bentoml_impl.frameworks.xgboost"
    )
    lightgbm = _LazyLoader(
        "bentoml.lightgbm", globals(), "_bentoml_impl.frameworks.lightgbm"
    )
    unsloth = _LazyLoader(
        "bentoml.unsloth", globals(), "_bentoml_impl.frameworks.unsloth"
    )
    mlflow = _LazyLoader("bentoml.mlflow", globals(), "_bentoml_impl.frameworks.mlflow")
    detectron = _LazyLoader(
        "bentoml.detectron",
        globals(),
        "bentoml._internal.frameworks.detectron",
        warning="`bentoml.detectron` is deprecated since v1.4 and will be removed in a future version.",
    )
    diffusers = _LazyLoader(
        "bentoml.diffusers",
        globals(),
        "bentoml._internal.frameworks.diffusers",
        warning="`bentoml.diffusers` is deprecated since v1.4 and will be removed in a future version.",
    )
    diffusers_simple = _LazyLoader(
        "bentoml.diffusers_simple",
        globals(),
        "bentoml.diffusers_simple",
        warning="`bentoml.diffusers_simple` is deprecated since v1.4 and will be removed in a future version.",
    )
    easyocr = _LazyLoader(
        "bentoml.easyocr",
        globals(),
        "bentoml._internal.frameworks.easyocr",
        warning="`bentoml.easyocr` is deprecated since v1.4 and will be removed in a future version.",
    )
    flax = _LazyLoader(
        "bentoml.flax",
        globals(),
        "bentoml._internal.frameworks.flax",
        warning="`bentoml.flax` is deprecated since v1.4 and will be removed in a future version.",
    )
    fastai = _LazyLoader(
        "bentoml.fastai",
        globals(),
        "bentoml._internal.frameworks.fastai",
        warning="`bentoml.fastai` is deprecated since v1.4 and will be removed in a future version.",
    )

    onnx = _LazyLoader(
        "bentoml.onnx",
        globals(),
        "bentoml._internal.frameworks.onnx",
        warning="`bentoml.onnx` is deprecated since v1.4 and will be removed in a future version.",
    )
    keras = _LazyLoader(
        "bentoml.keras",
        globals(),
        "bentoml._internal.frameworks.keras",
        warning="`bentoml.keras` is deprecated since v1.4 and will be removed in a future version.",
    )
    pytorch = _LazyLoader(
        "bentoml.pytorch",
        globals(),
        "bentoml._internal.frameworks.pytorch",
        warning="`bentoml.pytorch` is deprecated since v1.4 and will be removed in a future version.",
    )
    pytorch_lightning = _LazyLoader(
        "bentoml.pytorch_lightning",
        globals(),
        "bentoml._internal.frameworks.pytorch_lightning",
        warning="`bentoml.pytorch_lightning` is deprecated since v1.4 and will be removed in a future version.",
    )
    picklable_model = _LazyLoader(
        "bentoml.picklable_model",
        globals(),
        "bentoml._internal.frameworks.picklable_model",
        warning="`bentoml.picklable_model` is deprecated since v1.4 and will be removed in a future version.",
    )
    tensorflow = _LazyLoader(
        "bentoml.tensorflow",
        globals(),
        "bentoml._internal.frameworks.tensorflow",
        warning="`bentoml.tensorflow` is deprecated since v1.4 and will be removed in a future version.",
    )
    torchscript = _LazyLoader(
        "bentoml.torchscript",
        globals(),
        "bentoml._internal.frameworks.torchscript",
        warning="`bentoml.torchscript` is deprecated since v1.4 and will be removed in a future version.",
    )
    transformers = _LazyLoader(
        "bentoml.transformers",
        globals(),
        "bentoml._internal.frameworks.transformers",
        warning="`bentoml.transformers` is deprecated since v1.4 and will be removed in a future version.",
    )

    # Integrations
    triton = _LazyLoader("bentoml.triton", globals(), "bentoml.triton")
    ray = _LazyLoader("bentoml.ray", globals(), "bentoml.ray")
    gradio = _LazyLoader("bentoml.gradio", globals(), "bentoml.gradio")

    io = _LazyLoader("bentoml.io", globals(), "bentoml.io")
    batch = _LazyLoader("bentoml.batch", globals(), "bentoml.batch")
    models = _LazyLoader("bentoml.models", globals(), "bentoml.models")
    metrics = _LazyLoader("bentoml.metrics", globals(), "bentoml.metrics")
    container = _LazyLoader("bentoml.container", globals(), "bentoml.container")
    images = _LazyLoader("bentoml.images", globals(), "bentoml.images")
    client = _LazyLoader("bentoml.client", globals(), "bentoml.client")
    server = _LazyLoader("bentoml.server", globals(), "bentoml.server")
    exceptions = _LazyLoader("bentoml.exceptions", globals(), "bentoml.exceptions")
    monitoring = _LazyLoader("bentoml.monitoring", globals(), "bentoml.monitoring")
    cloud = _LazyLoader("bentoml.cloud", globals(), "bentoml.cloud")
    deployment = _LazyLoader("bentoml.deployment", globals(), "bentoml.deployment")
    validators = _LazyLoader("bentoml.validators", globals(), "bentoml.validators")
    del _LazyLoader, FrameworkImporter

    _NEW_SDK_ATTRS = [
        "service",
        "runner_service",
        "api",
        "task",
        "depends",
        "on_shutdown",
        "on_deployment",
        "asgi_app",
        "mount_asgi_app",
        "get_current_service",
        "IODescriptor",
    ]
    _NEW_CLIENTS = ["SyncHTTPClient", "AsyncHTTPClient"]

    if (ver := pkg_version_info("pydantic")) >= (2,):
        import _bentoml_sdk
    else:
        _bentoml_sdk = None

    def __getattr__(name: str) -> Any:
        if name in ("HTTPServer", "GrpcServer"):
            return getattr(server, name)

        if _bentoml_sdk is None:
            raise ImportError(
                f"The new SDK runs on pydantic>=2.0.0, but the you have {'.'.join(map(str, ver))}. "
                "Please upgrade it."
            )
        if name == "importing":
            from _bentoml_impl.loader import importing

            return importing
        elif name in _NEW_CLIENTS:
            import _bentoml_impl.client

            return getattr(_bentoml_impl.client, name)
        elif name in _NEW_SDK_ATTRS:
            return getattr(_bentoml_sdk, name)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "Context",
    "Cookie",
    "Service",
    "models",
    "batch",
    "metrics",
    "container",
    "server_context",
    "client",
    "io",
    "Tag",
    "Model",
    "Runner",
    "Runnable",
    "monitoring",
    "BentoCloudClient",  # BentoCloud REST API Client
    # bento APIs
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "build",
    "load",
    "push",
    "pull",
    "serve",
    "Bento",
    "exceptions",
    # Framework specific modules
    "catboost",
    "detectron",
    "unsloth",
    "diffusers",
    "diffusers_simple",
    "easyocr",
    "flax",
    "fastai",
    "lightgbm",
    "mlflow",
    "onnx",
    "picklable_model",
    "pytorch",
    "pytorch_lightning",
    "keras",
    "sklearn",
    "tensorflow",
    "torchscript",
    "transformers",
    "xgboost",
    # integrations
    "ray",
    "gradio",
    "cloud",
    "deployment",
    "triton",
    "monitor",
    "load_config",
    "save_config",
    "set_serialization_strategy",
    "Strategy",
    "Resource",
    # new SDK
    "service",
    "runner_service",
    "api",
    "task",
    "images",
    "on_shutdown",
    "on_deployment",
    "depends",
    "IODescriptor",
    "validators",
    "Field",
    "get_current_service",
    "asgi_app",
    "mount_asgi_app",
    # new implementation
    "SyncHTTPClient",
    "AsyncHTTPClient",
    "importing",
]
