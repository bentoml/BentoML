import typing as t
from pathlib import Path

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException
from ._internal.types import PathType

try:
    from catboost.core import CatBoost, Pool
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
            :obj:`catboost` is required by CatBoostModel
        InvalidArgument:
            model is not an instance of :class:`~catboost.core.CatBoost`

    Example usage under :code:`train.py`::

        TODO

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

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
        name: t.Optional[str] = "catboostmodel",
    ):
        if model:
            _model = model
        else:
            _model = self._model_type(model_type=model_type)
        super(CatBoostModel, self).__init__(_model, metadata=metadata, name=name)
        self._model_export_parameters = model_export_parameters
        self._model_pool: "Pool" = model_pool

    @classmethod
    def _model_type(cls, model_type: t.Optional[str] = "classifier"):
        try:
            from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
        except ImportError:
            raise MissingDependencyException("catboost is required by CatBoostModel")
        if model_type == "classifier":
            _model = CatBoostClassifier()
        elif model_type == "regressor":
            _model = CatBoostRegressor()
        else:
            _model = CatBoost()

        return _model

    def save(self, path: PathType) -> None:
        self._model.save_model(
            self.model_path(path, self.CATBOOST_FILE_EXTENSION),
            format=self.CATBOOST_FILE_EXTENSION.split('.')[1],
            export_parameters=self._model_export_parameters,
            pool=self._model_pool,
        )

    @classmethod
    def load(
        cls, path: PathType, model_type: t.Optional[str] = "classifier"
    ) -> CatBoost:
        model = cls._model_type(model_type=model_type)
        model_path: Path = cls.get_path(path, cls.CATBOOST_FILE_EXTENSION)
        if not model_path:
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.CATBOOST_FILE_EXTENSION} object."
            )
        model.load_model(str(model_path))
        return model
