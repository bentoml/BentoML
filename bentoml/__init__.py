# flake8: noqa: E402
from typing import TYPE_CHECKING

from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_global_config

# Inject dependencies and configurations
load_global_config()

from ._internal.log import configure_logging  # noqa: E402

configure_logging()

from . import models

# bento APIs are top-level
from .bentos import get
from .bentos import list  # pylint: disable=W0622
from .bentos import build
from .bentos import delete
from .bentos import export_bento
from .bentos import import_bento
from ._internal.types import Tag
from ._internal.utils import LazyLoader as _LazyLoader
from ._internal.service import Service
from ._internal.service.loader import load

if TYPE_CHECKING:
    from bentoml import catboost
    from bentoml import detectron
    from bentoml import easyocr
    from bentoml import flax
    from bentoml import gluon
    from bentoml import h2o
    from bentoml import mlflow
    from bentoml import lightgbm
    from bentoml import mlflow
    from bentoml import onnx
    from bentoml import onnxmlir
    from bentoml import keras
    from bentoml import paddle
    from bentoml import pycaret
    from bentoml import pytorch
    from bentoml import pytorch_lightning
    from bentoml import sklearn
    from bentoml import statsmodels
    from bentoml import tensorflow
    from bentoml import transformers
    from bentoml import xgboost
else:
    catboost = _LazyLoader("bentoml.catboost", globals(), "bentoml.catboost")
    detectron = _LazyLoader("bentoml.detectron", globals(), "bentoml.detectron")
    easyocr = _LazyLoader("bentoml.easyocr", globals(), "bentoml.easyocr")
    flax = _LazyLoader("bentoml.flax", globals(), "bentoml.flax")
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
    pytorch_lightning = _LazyLoader("bentoml.pytorch_lightning", globals(), "bentoml.pytorch_lightning")
    sklearn = _LazyLoader("bentoml.sklearn", globals(), "bentoml.sklearn")
    statsmodels = _LazyLoader("bentoml.statsmodels", globals(), "bentoml.statsmodels")
    tensorflow = _LazyLoader("bentoml.tensorflow", globals(), "bentoml.tensorflow")
    transformers = _LazyLoader("bentoml.transformers", globals(), "bentoml.transformers")
    xgboost = _LazyLoader("bentoml.xgboost", globals(), "bentoml.xgboost")

__all__ = [
    "__version__",
    "Service",
    "models",
    "Tag",
    # bento APIs
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "build",
    "load",
    # frameworks
    "catboost",
    "detectron",
    "easyocr",
    "flax",
    "gluon",
    "h2o",
    "lightgbm",
    "mlflow",
    "onnx",
    "onnxmlir",
    "paddle",
    "pycaret",
    "pytorch",
    "pytorch_lightning",
    "keras",
    "sklearn",
    "statsmodels",
    "tensorflow",
    "transformers",
    "xgboost",
]
