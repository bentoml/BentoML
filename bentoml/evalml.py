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

import pathlib
import typing as t

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import evalml
    from evalml.pipelines import PipelineBase
except ImportError:
    raise MissingDependencyException("evalml is required by EvalMLModel")


class EvalMLModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`evalml` models

    Args:
        model (`evalml.pipelines.PipelineBase`):
            Base pipeline for all EvalML model
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata
        name (`str`, `optional`, default to `evalmlmodel`):
            EvalMLModel instance name

    Raises:
        MissingDependencyException:
            :obj:`evalml` is required by EvalMLModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
        self,
        model: "evalml.pipelines.PipelineBase",
        metadata: t.Optional[MetadataType] = None,
        name: t.Optional[str] = "evalmlmodel",
    ):
        super(EvalMLModel, self).__init__(model, metadata=metadata, name=name)

    @classmethod
    def load(cls, path: PathType) -> "evalml.pipelines.PipelineBase":
        model_file_path: pathlib.Path = cls.get_path(path, cls.PICKLE_FILE_EXTENSION)
        return PipelineBase.load(str(model_file_path))

    def save(self, path: PathType) -> None:
        self._model.save(self.model_path(path, self.PICKLE_FILE_EXTENSION))
