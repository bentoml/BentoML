import os
import typing as t

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from .exceptions import InvalidArgument, MissingDependencyException

try:
    import catboost
    from catboost.core import CatBoost, CatBoostClassifier, CatBoostRegressor
except ImportError:
    raise MissingDependencyException("catboost is required by CatBoostModel")

CatBoostType = t.Union[CatBoost, CatBoostClassifier, CatBoostRegressor]


class CatBoostModel(Model):
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

    One then can define :code:`bento.py`::

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
            _model = self.__init_model_type(model_type=model_type)
        super(CatBoostModel, self).__init__(_model, metadata=metadata)
        self._model_export_parameters = model_export_parameters
        self._model_pool = model_pool

    @staticmethod
    def __init_model_type(model_type: t.Optional[str] = "classifier") -> CatBoostType:
        if model_type == "classifier":
            _model = CatBoostClassifier()
        elif model_type == "regressor":
            _model = CatBoostRegressor()
        else:
            _model = CatBoost()

        return _model

    def save(self, path: PathType) -> None:
        self._model.save_model(
            os.path.join(path, f"{MODEL_NAMESPACE}{self.CATBOOST_EXTENSION}"),
            format=self.CATBOOST_EXTENSION.split(".")[1],
            export_parameters=self._model_export_parameters,
            pool=self._model_pool,
        )

    @classmethod
    def load(  # noqa # pylint: disable=arguments-differ
        cls, path: PathType, model_type: t.Optional[str] = "classifier"
    ) -> "catboost.core.CatBoost":
        model = cls.__init_model_type(model_type=model_type)
        model_path: str = os.path.join(
            path, f"{MODEL_NAMESPACE}{cls.CATBOOST_EXTENSION}"
        )
        if not os.path.exists(model_path):
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.CATBOOST_EXTENSION} object."
            )
        model.load_model(model_path)
        return model
