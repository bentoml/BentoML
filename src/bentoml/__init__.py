# pylint: skip-file
"""
BentoML
=======

BentoML is the unified ML Model Serving framework. Data Scientists and ML Engineers use
BentoML to:

* Accelerate and standardize the process of taking ML models to production across teams
* Build reliable, scalable, and high performance model serving systems
* Provide a flexible MLOps platform that grows with your Data Science needs

To learn more, visit BentoML documentation at: http://docs.bentoml.org
To get involved with the development, find us on GitHub: https://github.com/bentoml
And join us in the BentoML slack community: https://l.linklyhq.com/l/ktOh
"""

from typing import TYPE_CHECKING

from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_global_config

# Inject dependencies and configurations
load_global_config()

# Bento management APIs
from .bentos import get
from .bentos import list  # pylint: disable=W0622
from .bentos import pull
from .bentos import push
from .bentos import delete
from .bentos import export_bento
from .bentos import import_bento

# BentoML built-in types
from ._internal.tag import Tag
from ._internal.bento import Bento
from ._internal.models import Model
from ._internal.runner import Runner
from ._internal.runner import Runnable
from ._internal.context import InferenceApiContext as Context
from ._internal.service import Service
from ._internal.utils.http import Cookie
from ._internal.yatai_client import YataiClient
from ._internal.monitoring.api import monitor
from ._internal.service.loader import load

# Framework specific modules, model management and IO APIs are lazily loaded upon import.
if TYPE_CHECKING:
    from . import h2o
    from . import flax
    from . import onnx
    from . import gluon
    from . import keras
    from . import spacy
    from . import fastai
    from . import mlflow
    from . import paddle
    from . import easyocr
    from . import pycaret
    from . import pytorch
    from . import sklearn
    from . import xgboost
    from . import catboost
    from . import lightgbm
    from . import onnxmlir
    from . import detectron
    from . import tensorflow
    from . import statsmodels
    from . import torchscript
    from . import transformers
    from . import tensorflow_v1
    from . import picklable_model
    from . import pytorch_lightning

    # isort: off

    from . import io
    from . import models

    # Prometheus metrics client
    from . import metrics

    # isort: on
else:
    from ._internal.utils import LazyLoader as _LazyLoader

    catboost = _LazyLoader("bentoml.catboost", globals(), "bentoml.catboost")
    detectron = _LazyLoader("bentoml.detectron", globals(), "bentoml.detectron")
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

    io = _LazyLoader("bentoml.io", globals(), "bentoml.io")
    models = _LazyLoader("bentoml.models", globals(), "bentoml.models")
    metrics = _LazyLoader("bentoml.metrics", globals(), "bentoml.metrics")

    del _LazyLoader

__all__ = [
    "__version__",
    "Context",
    "Cookie",
    "Service",
    "models",
    "metrics",
    "io",
    "Tag",
    "Model",
    "Runner",
    "Runnable",
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
    "Bento",
    # Framework specific modules
    "catboost",
    "detectron",
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
    "monitor",
]
