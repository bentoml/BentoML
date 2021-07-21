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

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import pyspark
except ImportError:
    raise MissingDependencyException('pyspark is required by PysparkModel')


class PysparkModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`pyspark` models.

    Args:
        model ():
            TODO:
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`pyspark` is required by PysparkModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(self, model, metadata: t.Optional[MetadataType] = None):
        super(PysparkModel, self).__init__(model, metadata=metadata)
        print(pyspark.F)

    def save(self, path: PathType) -> None:
        print(os.path.join(path))

    @classmethod
    def load(cls, path: PathType):
        pass
