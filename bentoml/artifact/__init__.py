# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bentoml.artifact.artifact import (
    BentoServiceArtifact,
    BentoServiceArtifactWrapper,
    ArtifactCollection,
)
from bentoml.artifact.text_file_artifact import TextFileArtifact
from bentoml.artifact.json_artifact import JSONArtifact
from bentoml.artifact.pickle_artifact import PickleArtifact
from bentoml.artifact.pytorch_model_artifact import PytorchModelArtifact
from bentoml.artifact.keras_model_artifact import KerasModelArtifact
from bentoml.artifact.xgboost_model_artifact import XgboostModelArtifact
from bentoml.artifact.h2o_model_artifact import H2oModelArtifact
from bentoml.artifact.fastai_model_artifact import FastaiModelArtifact
from bentoml.artifact.fastai2_model_artifact import Fastai2ModelArtifact
from bentoml.artifact.sklearn_model_artifact import SklearnModelArtifact
from bentoml.artifact.tf_savedmodel_artifact import TensorflowSavedModelArtifact
from bentoml.artifact.lightgbm_model_artifact import LightGBMModelArtifact
from bentoml.artifact.fasttext_model_artifact import FasttextModelArtifact
from bentoml.artifact.onnx_model_artifact import OnnxModelArtifact
from bentoml.artifact.spacy_model_artifact import SpacyModelArtifact
from bentoml.artifact.coreml_model_artifact import CoreMLModelArtifact

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
