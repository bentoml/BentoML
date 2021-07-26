import os
import typing as t

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import xgboost
except ImportError:
    raise MissingDependencyException("xgboost is required by XgBoostModel")


class XgBoostModel(Model):
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

    One then can define :code:`bento.py`::

        TODO:

    """

    XGBOOST_EXTENSION = ".model"

    def __init__(
        self, model: xgboost.core.Booster, metadata: t.Optional[MetadataType] = None
    ):
        super(XgBoostModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "xgboost.core.Booster":
        bst = xgboost.Booster()
        bst.load_model(os.path.join(path, f"{MODEL_NAMESPACE}{cls.XGBOOST_EXTENSION}"))
        return bst

    def save(self, path: PathType) -> None:
        self._model.save_model(
            os.path.join(path, f"{MODEL_NAMESPACE}{self.XGBOOST_EXTENSION}")
        )