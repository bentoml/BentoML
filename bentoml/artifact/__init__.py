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
from bentoml.utils.lazy_loader import LazyLoader

TextFileArtifact = LazyLoader(
    'TextFileArtifact',
    globals(),
    'bentoml.artifact.text_file_artifact.TextFileArtifact',
)
JSONArtifact = LazyLoader(
    'JSONArtifact', globals(), 'bentoml.artifact.json_artifact.JSONArtifact'
)
PickleArtifact = LazyLoader(
    'PickleArtifact', globals(), 'bentoml.artifact.pickle_artifact.PickleArtifact'
)
PytorchModelArtifact = LazyLoader(
    'PytorchModelArtifact',
    globals(),
    'bentoml.artifact.pytorch_model_artifact.PytorchModelArtifact',
)
KerasModelArtifact = LazyLoader(
    'KerasModelArtifact',
    globals(),
    'bentoml.artifact.keras_model_artifact.KerasModelArtifact',
)
XgboostModelArtifact = LazyLoader(
    'XgboostModelArtifact',
    globals(),
    'bentoml.artifact.xgboost_model_artifact.XgboostModelArtifact',
)
H2oModelArtifact = LazyLoader(
    'H2oModelArtifact',
    globals(),
    'bentoml.artifact.h2o_model_artifact.H2oModelArtifact',
)
FastaiModelArtifact = LazyLoader(
    'FastaiModelArtifact',
    globals(),
    'bentoml.artifact.fastai_model_artifact.FastaiModelArtifact',
)
Fastai2ModelArtifact = LazyLoader(
    'Fastai2ModelArtifact',
    globals(),
    'bentoml.artifact.fastai2_model_artifact.Fastai2ModelArtifact',
)
SklearnModelArtifact = LazyLoader(
    'SklearnModelArtifact',
    globals(),
    'bentoml.artifact.sklearn_model_artifact.SklearnModelArtifact',
)
TensorflowSavedModelArtifact = LazyLoader(
    'TensorflowSavedModelArtifact',
    globals(),
    'bentoml.artifact.tf_savedmodel_artifact.TensorflowSavedModelArtifact',
)
LightGBMModelArtifact = LazyLoader(
    'LightGBMModelArtifact',
    globals(),
    'bentoml.artifact.lightgbm_model_artifact.LightGBMModelArtifact',
)
FasttextModelArtifact = LazyLoader(
    'FasttextModelArtifact',
    globals(),
    'bentoml.artifact.fasttext_model_artifact.FasttextModelArtifact',
)
OnnxModelArtifact = LazyLoader(
    'OnnxModelArtifact',
    globals(),
    'bentoml.artifact.onnx_model_artifact.OnnxModelArtifact',
)
SpacyModelArtifact = LazyLoader(
    'SpacyModelArtifact',
    globals(),
    'bentoml.artifact.spacy_model_artifact.SpacyModelArtifact',
)


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
]
