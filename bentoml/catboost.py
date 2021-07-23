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
from ._internal.exceptions import InvalidArgument, MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import catboost
    from catboost.core import CatBoost, CatBoostClassifier, CatBoostRegressor
except ImportError:
    raise MissingDependencyException("catboost is required by CatBoostModel")


class CatBoostModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`catboost` model.

    Args:
        model (`catboost.core.CatBoost`, `optional`, default to `None`):
            CatBoost model
        model_type (`str`, `optional`, default to `classifier`):
            CatBoost model type. Options: ["classifier", "regressor", ""]
        model_export_parameters (`Dict[str, Any]`, `optional`, default to `None`):
            Additional format-dependent parameters.
        model_pool (`catboost.core.Pool`, `optional`, default to `None`):
            Dataset previously used for training, refers to
            ``catboost.core.Pools`` for more info.
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`catboost` is required by CatBoostModel
        InvalidArgument:
            model is not an instance of
            :class:`~catboost.core.CatBoost`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    CATBOOST_EXTENSION = ".cbm"
    _model: catboost.core.CatBoost

    def __init__(
        self,
        model: t.Optional["catboost.core.CatBoost"] = None,
        model_type: t.Optional[str] = "classifier",
        model_export_parameters: t.Optional[t.Dict[str, t.Any]] = None,
        model_pool: t.Optional["catboost.core.Pool"] = None,
        metadata: t.Optional[MetadataType] = None,
    ):
        if model:
            _model = model
        else:
            _model = self.__model__type(model_type=model_type)
        super(CatBoostModel, self).__init__(_model, metadata=metadata)
        self._model_export_parameters = model_export_parameters
        self._model_pool: "Pool" = model_pool

    @classmethod
    def __model__type(cls, model_type: t.Optional[str] = "classifier"):
        if model_type == "classifier":
            _model = CatBoostClassifier()
        elif model_type == "regressor":
            _model = CatBoostRegressor()
        else:
            _model = CatBoost()

        return _model

    def save(self, path: PathType) -> None:
        self._model.save_model(
            self.get_path(path, self.CATBOOST_EXTENSION),
            format=self.CATBOOST_EXTENSION.split(".")[1],
            export_parameters=self._model_export_parameters,
            pool=self._model_pool,
        )

    # fmt: off
    @classmethod
    def load(cls, path: PathType, model_type: t.Optional[str] = "classifier") -> "catboost.core.CatBoost":  # noqa # pylint: disable=arguments-differ
        # fmt: on
        model = cls.__model__type(model_type=model_type)
        get_path: str = cls.get_path(path, cls.CATBOOST_EXTENSION)
        if not os.path.exists(get_path):
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.CATBOOST_EXTENSION} object."
            )
        model.load_model(get_path)
        return model
