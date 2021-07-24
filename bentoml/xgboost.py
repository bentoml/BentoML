import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import xgboost
except ImportError:
    raise MissingDependencyException("xgboost is required by XgBoostModel")


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

    XGBOOST_EXTENSION = ".model"

    def __init__(
        self, model: xgboost.core.Booster, metadata: t.Optional[MetadataType] = None
    ):
        self._model: xgboost.core.Booster
        super(XgBoostModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "xgboost.core.Booster":
        bst = xgboost.Booster()
        bst.load_model(cls.get_path(path, cls.XGBOOST_EXTENSION))
        return bst

    def save(self, path: PathType) -> None:
        return self._model.save_model(self.get_path(path, self.XGBOOST_EXTENSION))
