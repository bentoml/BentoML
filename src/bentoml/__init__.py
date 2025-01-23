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

MODULE_ATTRS = {
    "Field": "pydantic:Field",
    # BentoML built-in types
    "load_config": "._internal.configuration:load_config",
    "save_config": "._internal.configuration:save_config",
    "set_serialization_strategy": "._internal.configuration:set_serialization_strategy",
    "Bento": "._internal.bento:Bento",
    "BentoCloudClient": "._internal.cloud:BentoCloudClient",
    "Context": "._internal.context:ServiceContext",
    "server_context": "._internal.context:server_context",
    "Model": "._internal.models:Model",
    "monitor": "._internal.monitoring:monitor",
    "Resource": "._internal.resource:Resource",
    "Runnable": "._internal.runner:Runnable",
    "Runner": "._internal.runner:Runner",
    "Strategy": "._internal.runner.strategy:Strategy",
    "Service": "._internal.service:Service",
    "Tag": "._internal.tag:Tag",
    "load": "._internal.service.loader:load",
    "Cookie": "._internal.utils.http:Cookie",
    # Bento management APIs
    "get": ".bentos:get",
    "build": ".bentos:build",
    "delete": ".bentos:delete",
    "export_bento": ".bentos:export_bento",
    "import_bento": ".bentos:import_bento",
    "list": ".bentos:list",
    "pull": ".bentos:pull",
    "push": ".bentos:push",
    "serve": ".bentos:serve",
    # Legacy APIs
    "HTTPServer": ".server:HTTPServer",
    "GrpcServer": ".server:GrpcServer",
    # New SDK
    "service": "_bentoml_sdk:service",
    "runner_service": "_bentoml_sdk:runner_service",
    "api": "_bentoml_sdk:api",
    "task": "_bentoml_sdk:task",
    "depends": "_bentoml_sdk:depends",
    "on_shutdown": "_bentoml_sdk:on_shutdown",
    "on_startup": "_bentoml_sdk:on_startup",
    "on_deployment": "_bentoml_sdk:on_deployment",
    "asgi_app": "_bentoml_sdk:asgi_app",
    "mount_asgi_app": "_bentoml_sdk:mount_asgi_app",
    "get_current_service": "_bentoml_sdk:get_current_service",
    "IODescriptor": "_bentoml_sdk:IODescriptor",
    "importing": "_bentoml_impl.loader:importing",
    "SyncHTTPClient": "_bentoml_impl.client:SyncHTTPClient",
    "AsyncHTTPClient": "_bentoml_impl.client:AsyncHTTPClient",
}


if TYPE_CHECKING:
    # Framework specific modules
    from pydantic import Field

    from _bentoml_impl.frameworks import catboost
    from _bentoml_impl.frameworks import lightgbm
    from _bentoml_impl.frameworks import mlflow
    from _bentoml_impl.frameworks import sklearn
    from _bentoml_impl.frameworks import xgboost

    # BentoML built-in types
    from ._internal.bento import Bento
    from ._internal.cloud import BentoCloudClient
    from ._internal.configuration import load_config
    from ._internal.configuration import save_config
    from ._internal.configuration import set_serialization_strategy
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
    from _bentoml_sdk import on_startup
    from _bentoml_sdk import runner_service
    from _bentoml_sdk import service
    from _bentoml_sdk import task
else:
    from _bentoml_impl.frameworks.importer import FrameworkImporter

    from ._internal.utils.lazy_loader import LazyLoader as _LazyLoader

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

    def __getattr__(name: str) -> Any:
        if name in MODULE_ATTRS:
            from importlib import import_module

            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        raise AttributeError(f"module {__name__} has no attribute {name}")


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
    "on_startup",
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
