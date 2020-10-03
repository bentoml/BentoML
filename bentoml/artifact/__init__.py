import logging

logger = logging.getLogger(__name__)
logger.warning(
    """\
Importing from "bentoml.artifact.*" has been deprecated. Instead, use\
`bentoml.frameworks.*` and `bentoml.service.*`. e.g.:, \
`from bentoml.frameworks.sklearn import SklearnModelArtifact`, \
`from bentoml.service.artifacts import BentoServiceArtifact`, \
`from bentoml.service.artifacts.common import PickleArtifact`"""
)

from bentoml.service.artifacts import (
    BentoServiceArtifact,
    BentoServiceArtifactWrapper,
    ArtifactCollection,
)

from bentoml.service.artifacts.common import TextFileArtifact
from bentoml.service.artifacts.common import JSONArtifact
from bentoml.service.artifacts.common import PickleArtifact

from bentoml.frameworks.coreml import CoreMLModelArtifact
from bentoml.frameworks.fastai import FastaiModelArtifact
from bentoml.frameworks.fastai2 import Fastai2ModelArtifact
from bentoml.frameworks.fasttext import FasttextModelArtifact
from bentoml.frameworks.h2o import H2oModelArtifact
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.frameworks.lightgbm import LightGBMModelArtifact
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.frameworks.spacy import SpacyModelArtifact
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.frameworks.xgboost import XgboostModelArtifact

__all__ = [
    "ArtifactCollection",
    "BentoServiceArtifact",
    "BentoServiceArtifactWrapper",
    "CoreMLModelArtifact",
    "Fastai2ModelArtifact",
    "FastaiModelArtifact",
    "FasttextModelArtifact",
    "H2oModelArtifact",
    "JSONArtifact",
    "KerasModelArtifact",
    "LightGBMModelArtifact",
    "OnnxModelArtifact",
    "PickleArtifact",
    "PytorchModelArtifact",
    "SklearnModelArtifact",
    "SpacyModelArtifact",
    "TensorflowSavedModelArtifact",
    "TextFileArtifact",
    "XgboostModelArtifact",
]
