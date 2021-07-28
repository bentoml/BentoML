import os
import typing as t

from ._internal.models.base import Model
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import h2o

    if t.TYPE_CHECKING:
        import h2o.model
except ImportError:
    raise MissingDependencyException("h2o is required by H2OModel")


class H2OModel(Model):
    """
    Model class for saving/loading :obj:`h2o` models
     using meth:`~h2o.saved_model` and :meth:`~h2o.load_model`

    Args:
        model (`h2o.model.model_base.ModelBase`):
            :obj:`ModelBase` for all h2o model instance
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`h2o` is required by H2OModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(
        self,
        model: "h2o.model.model_base.ModelBase",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(H2OModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "h2o.model.model_base.ModelBase":
        h2o.init()
        h2o.no_progress()
        # NOTE: model_path should be the first item in
        #   h2o saved artifact directory
        model_path: str = str(os.path.join(path, os.listdir(path)[0]))
        return h2o.load_model(model_path)

    def save(self, path: PathType) -> None:
        h2o.save_model(model=self._model, path=str(path), force=True)
