"""
BentoML
=======

BentoML is the unified ML Model Serving framework. Data Scientists and ML Engineers use
BentoML to:

* Accelerate and standardize the process of taking ML models to production across teams
* Build reliable, scalable, and high performance model serving systems
* Provide a flexible MLOps platform that grows with your Data Science needs

To learn more, visit BentoML documentation at: http://docs.bentoml.com
To get involved with the development, find us on GitHub: https://github.com/bentoml
And join us in the BentoML slack community: https://l.bentoml.com/join-slack
"""

from typing import TYPE_CHECKING
from typing import Any

from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_config
from ._internal.configuration import save_config
from ._internal.configuration import set_serialization_strategy

# Inject dependencies and configurations
load_config()

# BentoML built-in types
from ._internal.bento import Bento
from ._internal.cloud import YataiClient
from ._internal.context import ServiceContext as Context
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
from .bentos import delete
from .bentos import export_bento
from .bentos import get
from .bentos import import_bento
from .bentos import list  # pylint: disable=W0622
from .bentos import pull
from .bentos import push
from .bentos import serve

# server API
from .server import GrpcServer
from .server import HTTPServer

# Framework specific modules, model management and IO APIs are lazily loaded upon import.
if TYPE_CHECKING:
    from . import catboost
    from . import detectron
    from . import diffusers
    from . import diffusers_simple
    from . import easyocr
    from . import fastai
    from . import flax
    from . import gluon
    from . import h2o
    from . import keras
    from . import lightgbm
    from . import mlflow
    from . import onnx
    from . import onnxmlir
    from . import paddle
    from . import picklable_model
    from . import pycaret
    from . import pytorch
    from . import pytorch_lightning
    from . import ray
    from . import sklearn
    from . import spacy
    from . import statsmodels
    from . import tensorflow
    from . import tensorflow_v1
    from . import torchscript
    from . import transformers
    from . import triton
    from . import xgboost

    # isort: off
    from . import io
    from . import models
    from . import metrics  # Prometheus metrics client
    from . import container  # Container API
    from . import client  # Client API
    from . import batch  # Batch API
    from . import exceptions  # BentoML exceptions
    from . import server  # Server API
    from . import ui  # BentoML UI
    from . import monitoring  # Monitoring API
    from . import cloud  # Cloud API

    # isort: on
    from _bentoml_sdk import api
    from _bentoml_sdk import depends
    from _bentoml_sdk import runner_service
    from _bentoml_sdk import service
else:
    from ._internal.utils import LazyLoader as _LazyLoader

    # ML Frameworks
    catboost = _LazyLoader("bentoml.catboost", globals(), "bentoml.catboost")
    detectron = _LazyLoader("bentoml.detectron", globals(), "bentoml.detectron")
    diffusers = _LazyLoader("bentoml.diffusers", globals(), "bentoml.diffusers")
    diffusers_simple = _LazyLoader(
        "bentoml.diffusers_simple", globals(), "bentoml.diffusers_simple"
    )
    easyocr = _LazyLoader("bentoml.easyocr", globals(), "bentoml.easyocr")
    flax = _LazyLoader("bentoml.flax", globals(), "bentoml.flax")
    fastai = _LazyLoader("bentoml.fastai", globals(), "bentoml.fastai")
    gluon = _LazyLoader("bentoml.gluon", globals(), "bentoml.gluon")
    h2o = _LazyLoader("bentoml.h2o", globals(), "bentoml.h2o")
    lightgbm = _LazyLoader("bentoml.lightgbm", globals(), "bentoml.lightgbm")
    mlflow = _LazyLoader("bentoml.mlflow", globals(), "bentoml.mlflow")
    onnx = _LazyLoader("bentoml.onnx", globals(), "bentoml.onnx")
    onnxmlir = _LazyLoader("bentoml.onnxmlir", globals(), "bentoml.onnxmlir")
    keras = _LazyLoader("bentoml.keras", globals(), "bentoml.keras")
    paddle = _LazyLoader("bentoml.paddle", globals(), "bentoml.paddle")
    pycaret = _LazyLoader("bentoml.pycaret", globals(), "bentoml.pycaret")
    pytorch = _LazyLoader("bentoml.pytorch", globals(), "bentoml.pytorch")
    pytorch_lightning = _LazyLoader(
        "bentoml.pytorch_lightning", globals(), "bentoml.pytorch_lightning"
    )
    sklearn = _LazyLoader("bentoml.sklearn", globals(), "bentoml.sklearn")
    picklable_model = _LazyLoader(
        "bentoml.picklable_model", globals(), "bentoml.picklable_model"
    )
    spacy = _LazyLoader("bentoml.spacy", globals(), "bentoml.spacy")
    statsmodels = _LazyLoader("bentoml.statsmodels", globals(), "bentoml.statsmodels")
    tensorflow = _LazyLoader("bentoml.tensorflow", globals(), "bentoml.tensorflow")
    tensorflow_v1 = _LazyLoader(
        "bentoml.tensorflow_v1", globals(), "bentoml.tensorflow_v1"
    )
    torchscript = _LazyLoader("bentoml.torchscript", globals(), "bentoml.torchscript")
    transformers = _LazyLoader(
        "bentoml.transformers", globals(), "bentoml.transformers"
    )
    xgboost = _LazyLoader("bentoml.xgboost", globals(), "bentoml.xgboost")

    # Integrations
    triton = _LazyLoader("bentoml.triton", globals(), "bentoml.triton")
    ray = _LazyLoader("bentoml.ray", globals(), "bentoml.ray")

    io = _LazyLoader("bentoml.io", globals(), "bentoml.io")
    ui = _LazyLoader("bentoml.ui", globals(), "bentoml.ui")
    batch = _LazyLoader("bentoml.batch", globals(), "bentoml.batch")
    models = _LazyLoader("bentoml.models", globals(), "bentoml.models")
    metrics = _LazyLoader("bentoml.metrics", globals(), "bentoml.metrics")
    container = _LazyLoader("bentoml.container", globals(), "bentoml.container")
    client = _LazyLoader("bentoml.client", globals(), "bentoml.client")
    server = _LazyLoader("bentoml.server", globals(), "bentoml.server")
    exceptions = _LazyLoader("bentoml.exceptions", globals(), "bentoml.exceptions")
    monitoring = _LazyLoader("bentoml.monitoring", globals(), "bentoml.monitoring")
    cloud = _LazyLoader("bentoml.cloud", globals(), "bentoml.cloud")

    del _LazyLoader

    _NEW_SDK_ATTRS = ["service", "runner_service", "api", "depends"]

    def __getattr__(name: str) -> Any:
        if name not in _NEW_SDK_ATTRS:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        import _bentoml_sdk

        return getattr(_bentoml_sdk, name)


__all__ = [
    "__version__",
    "Context",
    "Cookie",
    "Service",
    "models",
    "batch",
    "metrics",
    "container",
    "client",
    "server",
    "io",
    "types",
    "Tag",
    "Model",
    "Runner",
    "Runnable",
    "monitoring",
    "YataiClient",  # Yatai REST API Client
    # bento APIs
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "load",
    "push",
    "pull",
    "serve",
    "Bento",
    "exceptions",
    # server APIs
    "HTTPServer",
    "GrpcServer",
    # Framework specific modules
    "catboost",
    "detectron",
    "diffusers",
    "diffusers_simple",
    "easyocr",
    "flax",
    "fastai",
    "gluon",
    "h2o",
    "lightgbm",
    "mlflow",
    "onnx",
    "onnxmlir",
    "paddle",
    "picklable_model",
    "pycaret",
    "pytorch",
    "pytorch_lightning",
    "keras",
    "sklearn",
    "spacy",
    "statsmodels",
    "tensorflow",
    "tensorflow_v1",
    "torchscript",
    "transformers",
    "xgboost",
    # integrations
    "ray",
    "cloud",
    "triton",
    "monitor",
    "load_config",
    "save_config",
    "set_serialization_strategy",
    "Strategy",
    "Resource",
    "service",
    "runner_service",
    "api",
    "depends",
]
