import typing as t
from pathlib import Path

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException
from ._internal.types import PathType

try:
    from catboost import CatBoost, Pool
except ImportError:
    raise MissingDependencyException("catboost is required by CatBoostModel")


class CatBoostModel(BaseArtifact):
    """
    Model class for saving/loading CatBoost model that can be used in a BentoML bundle.

    Class attributes:

    - model (`catboost.core.CatBoost`, `optional`):
        CatBoost model
    - model_type (`str`, `optional`, default to `classifier`):
        CatBoost model type. Options: ["classifier", "regressor", ""]
    - model_export_parameters (`Dict[str, Any]`, `optional`, default to `None`):
        Additional format-dependent parameters.
    - model_pool (`catboost.core.Pool`, `optional`, default to `None`):
        Dataset previously used for training, refers to catboost.core.Pools for more info.
    - metadata (`Dict[str, Any]`, `optional`, default to `None`):
        Class metadata

    Raises:
        MissingDependencyException:
        InvalidArgument:

    Example usage::
        TODO:
    """

    CATBOOST_FILE_EXTENSION = ".cbm"
    _model: CatBoost

    def __init__(
        self,
        model: t.Optional["CatBoost"] = None,
        model_type: t.Optional[str] = "classifier",
        model_export_parameters: t.Optional[t.Dict[str, t.Any]] = None,
        model_pool: t.Optional["Pool"] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        if model:
            _model = model
        else:
            try:
                from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
            except ImportError:
                raise MissingDependencyException(
                    "catboost is required by CatBoostModel"
                )
            if model_type == "classifier":
                _model = CatBoostClassifier()
            elif model_type == "regressor":
                _model = CatBoostRegressor()
            else:
                _model = CatBoost()
        super(CatBoostModel, self).__init__(_model, metadata=metadata)
        self._model_export_parameters = model_export_parameters
        self._model_pool: "Pool" = model_pool
        self.__name__ = "catboostmodel"

    def save(self, path: PathType) -> None:
        self._model.save_model(
            self.model_path(path, self.CATBOOST_FILE_EXTENSION),
            format=self.CATBOOST_FILE_EXTENSION,
            export_parameters=self._model_export_parameters,
            pool=self._model_pool,
        )

    @classmethod
    def load(cls, path: PathType) -> CatBoost:
        model = getattr(cls, "_model")
        model_path: Path = cls.ext_path(path, cls.CATBOOST_FILE_EXTENSION)
        if not model_path:
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.CATBOOST_FILE_EXTENSION} object."
            )
        model.load_model(cls.ext_path(path, cls.CATBOOST_FILE_EXTENSION))
        return model
