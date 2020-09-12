import logging

logger = logging.getLogger(__name__)
logger.warning(
    """\
Importing from "bentoml.artifact.*" has been deprecated. Instead, use\
`bentoml.frameworks.*` and `bentoml.service.*`. e.g.:, \
`from bentoml.frameworks.sklearn import SklearnModelArtifact`, \
`from bentoml.service.artifacts import BentoServiceArtifact`, \
`from bentoml.service.common_artifacts import PickleArtifact`"""
)

from bentoml.service.artifacts import (
    BentoServiceArtifact,
    BentoServiceArtifactWrapper,
    ArtifactCollection,
)
from bentoml.service.common_artifacts import (
    TextFileArtifact,
    JSONArtifact,
    PickleArtifact,
)
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.frameworks.xgboost import XgboostModelArtifact
from bentoml.frameworks.fastai import FastaiModelArtifact
from bentoml.frameworks.fastai2 import Fastai2ModelArtifact
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.frameworks.lightgbm import LightGBMModelArtifact
from bentoml.frameworks.fasttext import FasttextModelArtifact
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.frameworks.spacy import SpacyModelArtifact
from bentoml.frameworks.coreml import CoreMLModelArtifact
from bentoml.frameworks.h2o import H2oModelArtifact

__all__ = [
    "BentoServiceArtifact",
    "BentoServiceArtifactWrapper",
    "ArtifactCollection",
    "PickleArtifact",
    "PytorchModelArtifact",
    "TextFileArtifact",
    "JSONArtifact",
    "KerasModelArtifact",
    "XgboostModelArtifact",
    "H2oModelArtifact",
    "FastaiModelArtifact",
    "Fastai2ModelArtifact",
    "SklearnModelArtifact",
    "TensorflowSavedModelArtifact",
    "LightGBMModelArtifact",
    "FasttextModelArtifact",
    "OnnxModelArtifact",
    "SpacyModelArtifact",
    "CoreMLModelArtifact",
]
