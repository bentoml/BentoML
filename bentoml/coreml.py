import os
import typing as t

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader
from .exceptions import InvalidArgument

if t.TYPE_CHECKING:
    import coremltools as cmt  # pylint: disable=unused-import
    import coremltools.models as models  # pylint: disable=unused-import
else:
    cmt = LazyLoader("cmt", globals(), "coremltools")
    models = LazyLoader("models", globals(), "coremltools.models")


class CoreMLModel(Model):
    """
    Model class for saving/loading :obj:`coremltools.models.MLModel`
    model that can be used in a BentoML bundle.

    Args:
        model (`coremltools.models.MLModel`):
            :class:`~coreml.models.MLModel` instance
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`coremltools` is required by CoreMLModel
        InvalidArgument:
            model is not an instance of :class:`~coremltools.models.MLModel`

    Example usage under :code:`train.py`::

        TODO

    One then can define :code:`bento.py`::

        TODO:
    """

    if int(cmt.__version__.split(".")[0]) == 4:
        COREMLMODEL_EXTENSION = ".mlmodel"
    else:
        # for coremltools>=5.0
        COREMLMODEL_EXTENSION = ".mlpackage"
    _model: "cmt.models.MLModel"

    def __init__(
        self,
        model: "cmt.models.MLModel",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(CoreMLModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "cmt.models.MLModel":
        get_path: str = os.path.join(
            path, f"{MODEL_NAMESPACE}{cls.COREMLMODEL_EXTENSION}"
        )
        if not os.path.exists(get_path):
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.COREMLMODEL_EXTENSION}."
            )
        return models.MLModel(get_path)

    def save(self, path: PathType) -> None:
        self._model.save(
            os.path.join(path, f"{MODEL_NAMESPACE}{self.COREMLMODEL_EXTENSION}")
        )
