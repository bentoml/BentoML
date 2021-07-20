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

import os
import typing as t

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import coremltools
except ImportError:
    raise MissingDependencyException("coremltools>=4.0b2 is required by CoreMLModel")


class CoreMLModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`coremltools.models.MLModel`
    model that can be used in a BentoML bundle.

    Args:
        model (`coremltools.models.MLModel`):
            :class:`~coreml.models.MLModel` instance
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`coremltools` is required by CoreMLModel
        InvalidArgument:
            model is not an instance of :class:`~coremltools.models.MLModel`

    Example usage under :code:`train.py`::

        TODO

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    if int(coremltools.__version__.split(".")[0]) == 4:
        COREMLMODEL_FILE_EXTENSION = ".mlmodel"
    else:
        # for coremltools>=5.0
        COREMLMODEL_FILE_EXTENSION = ".mlpackage"
    _model: "coremltools.models.MLModel"

    def __init__(
        self,
        model: "coremltools.models.MLModel",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(CoreMLModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "coremltools.models.MLModel":
        model_path: str = cls.model_path(path, cls.COREMLMODEL_FILE_EXTENSION)
        if not os.path.exists(model_path):
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.COREMLMODEL_FILE_EXTENSION}."
            )
        return coremltools.models.MLModel(model_path)

    def save(self, path: PathType) -> None:
        self._model.save(self.model_path(path, self.COREMLMODEL_FILE_EXTENSION))
