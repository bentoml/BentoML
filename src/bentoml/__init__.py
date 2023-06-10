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
And join us in the BentoML slack community: https://l.bentoml.com/join-slack
"""
from __future__ import annotations

import typing as t

from ._internal.configuration import BENTOML_VERSION as __version__
from ._internal.configuration import load_config
from ._internal.configuration import save_config as save_config

# Inject dependencies and configurations
load_config()

from ._internal import utils as utils

_import_structure = {
    # NOTE: Bento management APIs.
    "bentos": [
        "get",
        "list",
        "pull",
        "push",
        "serve",
        "delete",
        "export_bento",
        "import_bento",
    ],
    # NOTE: server APIs.
    "server": ["GrpcServer", "HTTPServer"],
    # NOTE: BentoML built-in types
    "_internal.tag": ["Tag"],
    "_internal.bento": ["Bento"],
    "_internal.models": ["Model"],
    "_internal.runner": ["Runner", "Runnable"],
    "_internal.context": ["Context"],
    "_internal.service": ["Service"],
    "_internal.utils.http": ["Cookie"],
    "_internal.yatai_client": ["YataiClient"],
    "_internal.configuration": ["set_serialization_strategy"],
    "_internal.monitoring.api": ["monitor"],
    "_internal.service.loader": ["load"],
    # NOTE: Provisional import. For better ease-of-use.
    # A mixed of BentoML's and framework exports
    "batch": [],
    "catboost": [],
    "client": [],
    "container": [],
    "detectron": [],
    "diffusers": [],
    "easyocr": [],
    "evalml": [],
    "exceptions": [],
    "fastai": [],
    "fasttext": [],
    "flax": [],
    "gluon": [],
    "h2o": [],
    "io": [],
    "keras": [],
    "lightgbm": [],
    "metrics": [],
    "mlflow": [],
    "models": [],
    "monitoring": [],
    "onnx": [],
    "onnxmlir": [],
    "paddle": [],
    "picklable_model": [],
    "pycaret": [],
    "pyspark": [],
    "pytorch": [],
    "pytorch_lightning": [],
    "ray": [],
    "sklearn": [],
    "spacy": [],
    "statsmodels": [],
    "tensorflow": [],
    "tensorflow_v1": [],
    "torchscript": [],
    "transformers": [],
    "triton": [],
    "utils": [],
    "xgboost": [],
}


# NOTE: The actual import are lazily loaded. All imports defined under TYPE_CHECKING
# are for IDE and linter to be nice with.
if t.TYPE_CHECKING:
    # Framework imports
    from . import batch as batch  # Batch API
    from . import catboost as catboost
    from . import client as client  # Client API
    from . import container as container  # Container API
    from . import detectron as detectron
    from . import diffusers as diffusers
    from . import easyocr as easyocr
    from . import exceptions as exceptions  # BentoML exceptions
    from . import fastai as fastai
    from . import flax as flax
    from . import gluon as gluon
    from . import h2o as h2o
    from . import io as io
    from . import keras as keras
    from . import lightgbm as lightgbm
    from . import metrics as metrics  # Prometheus metrics client
    from . import mlflow as mlflow
    from . import models as models
    from . import onnx as onnx
    from . import onnxmlir as onnxmlir
    from . import paddle as paddle
    from . import picklable_model as picklable_model
    from . import pycaret as pycaret
    from . import pytorch as pytorch
    from . import pytorch_lightning as pytorch_lightning
    from . import ray as ray
    from . import server as server  # Server API
    from . import sklearn as sklearn
    from . import spacy as spacy
    from . import statsmodels as statsmodels
    from . import tensorflow as tensorflow
    from . import tensorflow_v1 as tensorflow_v1
    from . import torchscript as torchscript
    from . import transformers as transformers
    from . import triton as triton
    from . import xgboost as xgboost
    from ._internal.bento import Bento as Bento
    from ._internal.configuration import \
        set_serialization_strategy as set_serialization_strategy
    from ._internal.context import Context as Context
    from ._internal.models import Model as Model
    from ._internal.monitoring.api import monitor as monitor
    from ._internal.runner import Runnable as Runnable
    from ._internal.runner import Runner as Runner
    from ._internal.service import Service as Service
    from ._internal.service.loader import load as load
    # BentoML built-in types
    from ._internal.tag import Tag as Tag
    from ._internal.utils.http import Cookie as Cookie
    from ._internal.yatai_client import YataiClient as YataiClient
    # Bento management APIs
    from .bentos import delete as delete
    from .bentos import export_bento as export_bento
    from .bentos import get as get
    from .bentos import import_bento as import_bento
    from .bentos import list as list  # pylint: disable=W0622
    from .bentos import pull as pull
    from .bentos import push as push
    from .bentos import serve as serve
    # server API
    from .server import GrpcServer as GrpcServer
    from .server import HTTPServer as HTTPServer
else:
    import sys

    sys.modules[__name__] = utils.LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
