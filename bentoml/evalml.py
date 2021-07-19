# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import importlib
import os
import typing as t

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    importlib.import_module("evalml")
except ImportError:
    raise MissingDependencyException("evalml is required by EvalMLModel")


class EvalMLModel(BaseArtifact):
    """
    Artifact class  for saving/loading EvalML models

    Args:
        name (str): Name for the artifact

    Raises:
        MissingDependencyException:
            `evalml` is required by EvalMLModel

    Example usage::

    >>> # import the EvalML base pipeline class corresponding to your ML task
    >>> from evalml.pipelines import BinaryClassificationPipeline
    >>> # create an EvalML pipeline instance with desired components
    >>> model_to_save = BinaryClassificationPipeline(
    >>>     ['Imputer', 'One Hot Encoder', 'Random Forest Classifier'])
    >>> # Train model with data using model_to_save.fit(X, y)
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.sklearn import EvalMLModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([EvalMLModelArtifact('EvalML pipeline')])
    >>> class EvalMLModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         result = self.artifacts.model.predict(df)
    >>>         return result
    >>>
    >>> svc = EvalMLModelService()
    >>>
    >>> # Pack directly with EvalML model object
    >>> svc.pack('model', model_to_save)
    >>> svc.save()
    """

    def __init__(
        self,
        model,
        metadata: t.Optional[MetadataType] = None,
        name: t.Optional[str] = "evalmlmodel",
    ):
        super(EvalMLModel, self).__init__(model, metadata=metadata, name=name)

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self.PICKLE_FILE_EXTENSION)

    def load(self, path: PathType):
        self._validate_package()
        model_file_path = self._model_file_path(path)
        evalml_pipelines_module = importlib.import_module("evalml.pipelines")
        model = evalml_pipelines_module.PipelineBase.load(model_file_path)
        self.pack(model)
        return model

    def save(self, path):
        self._validate_package()
        model_file_path = self._model_file_path(path)
        self._model.save(model_file_path)
