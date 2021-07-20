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
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import lightgbm
except ImportError:
    raise MissingDependencyException("lightgbm is required by LightGBMModel")


class LightGBMModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`lightgbm` models

    Args:
        model (`lightgbm.Booster`):
            LightGBM model instance is of type :class:`lightgbm.Booster`
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`lightgbm` is required by LightGBMModel 

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
        self, model: "lightgbm.Booster", metadata: t.Optional[MetadataType] = None,
    ):
        super(LightGBMModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "lightgbm.Booster":
        return lightgbm.Booster(model_file=cls.model_path(path, cls.TXT_FILE_EXTENSION))

    def save(self, path: PathType) -> None:
        self._model.save_model(self.model_path(path, self.TXT_FILE_EXTENSION))
