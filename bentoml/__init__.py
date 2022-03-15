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
from .bentos import pull
from .bentos import push
from .bentos import build
from .bentos import delete
from .bentos import export_bento
from .bentos import import_bento
from ._internal.tag import Tag
from ._internal.utils import LazyLoader as _LazyLoader
from ._internal.models import Model
from ._internal.runner import Runner
from ._internal.runner import SimpleRunner
from ._internal.service import Service
from ._internal.yatai_client import YataiClient
from ._internal.service.loader import load

if TYPE_CHECKING:
    from bentoml import h2o
    from bentoml import flax
    from bentoml import onnx
    from bentoml import gluon
    from bentoml import keras
    from bentoml import spacy
    from bentoml import mlflow
    from bentoml import paddle
    from bentoml import easyocr
    from bentoml import pycaret
    from bentoml import pytorch
    from bentoml import sklearn
    from bentoml import xgboost
    from bentoml import catboost
    from bentoml import lightgbm
    from bentoml import onnxmlir
    from bentoml import detectron
    from bentoml import tensorflow
    from bentoml import statsmodels
    from bentoml import torchscript
    from bentoml import transformers
    from bentoml import tensorflow_v1
    from bentoml import picklable_model
    from bentoml import pytorch_lightning
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

__all__ = [
    "__version__",
    "Service",
    "models",
    "Tag",
    "Runner",
    "SimpleRunner",
    "YataiClient",
    "Model",
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
]
