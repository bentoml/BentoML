import os
import typing as t

from ._internal.models.base import JSON_EXTENSION, MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

if t.TYPE_CHECKING:
    import xgboost as xgb  # pylint: disable=unused-import
else:
    xgb = LazyLoader("xgb", globals(), "xgboost")


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

    def __init__(
        self, model: "xgb.core.Booster", metadata: t.Optional[MetadataType] = None
    ):
        super(XgBoostModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(  # noqa # pylint: disable=arguments-differ
        cls, path: PathType, infer_params: t.Dict[str, t.Union[str, int]] = None
    ) -> "xgb.core.Booster":
        return xgb.core.Booster(
            params=infer_params,
            model_file=os.path.join(path, f"{MODEL_NAMESPACE}{JSON_EXTENSION}"),
        )

    def save(self, path: PathType) -> None:
        self._model.save_model(os.path.join(path, f"{MODEL_NAMESPACE}{JSON_EXTENSION}"))
