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


class XgBoostModel(ModelArtifact):
    """
    Artifact class for saving/loading :obj:`xgboost` model

    Args:
        model (`xgboost.core.Booster`):
            Every xgboost model instance of type :obj:`xgboost.core.Booster`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`xgboost` is required by XgBoostModel
        TypeError:
           model must be instance of :obj:`xgboost.core.Booster`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    try:
        import xgboost
    except ImportError:
        raise MissingDependencyException("xgboost is required by XgBoostModel")

    XGBOOST_EXTENSION = ".model"
    _model: xgboost.core.Booster

    def __init__(
        self, model: xgboost.core.Booster, metadata: t.Optional[MetadataType] = None
    ):
        super(XgBoostModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "xgboost.core.Booster":
        try:
            import xgboost as xgb
        except ImportError:
            raise MissingDependencyException("xgboost is required by XgBoostModel")
        bst = xgb.Booster()
        bst.load_model(cls.get_path(path, cls.XGBOOST_EXTENSION))
        return bst

    def save(self, path: PathType) -> None:
        return self._model.save_model(self.get_path(path, self.XGBOOST_EXTENSION))
