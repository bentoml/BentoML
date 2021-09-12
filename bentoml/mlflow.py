import os
import typing as t

import bentoml._internal.constants as _const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import GenericDictType, PathType
from ._internal.utils import LazyLoader
from .exceptions import InvalidArgument

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="mlflow",
    module=__name__,
    inst="`pip install mlflow`",
)

MT = t.TypeVar("_MT")

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import mlflow
else:
    mlflow = LazyLoader("mlflow", globals(), "mlflow", exc_msg=_exc)


class MLflowModel(Model):
    """
    Model class for saving/loading :obj:`mlflow` models

    Args:
        model (`mlflow.models.Model`):
            All mlflow models are of type :obj:`mlflow.models.Model`
        loader_module (`types.ModuleType`):
            flavors supported by :obj:`mlflow`
        metadata (`GenericDictType`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`mlflow` is required by MLflowModel
        ArtifactLoadingException:
            given `loader_module` is not supported by :obj:`mlflow`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(
        self,
        model: MT,
        loader_module: t.Type["mlflow.pyfunc"],
        metadata: t.Optional[GenericDictType] = None,
    ):
        super(MLflowModel, self).__init__(model, metadata=metadata)
        if "mlflow" not in loader_module.__name__:
            raise InvalidArgument("given `loader_module` is not omitted by mlflow.")
        self._loader_module: t.Type["mlflow.pyfunc"] = loader_module

    @classmethod
    def load(cls, path: PathType) -> "mlflow.pyfunc.PyFuncModel":
        project_path: str = str(os.path.join(path, MODEL_NAMESPACE))
        return mlflow.pyfunc.load_model(project_path)

    def save(self, path: PathType) -> None:
        self._loader_module.save_model(self._model, os.path.join(path, MODEL_NAMESPACE))
