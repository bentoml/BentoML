import os
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import InvalidArgument, MissingDependencyException

try:
    import coremltools
    import coremltools.models
except ImportError:
    raise MissingDependencyException("coremltools>=4.0b2 is required by CoreMLModel")


class CoreMLModel(ModelArtifact):
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

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    if int(coremltools.__version__.split(".")[0]) == 4:
        COREMLMODEL_EXTENSION = ".mlmodel"
    else:
        # for coremltools>=5.0
        COREMLMODEL_EXTENSION = ".mlpackage"
    _model: "coremltools.models.MLModel"

    def __init__(
        self,
        model: "coremltools.models.MLModel",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(CoreMLModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "coremltools.models.MLModel":
        get_path: str = cls.get_path(path, cls.COREMLMODEL_EXTENSION)
        if not os.path.exists(get_path):
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.COREMLMODEL_EXTENSION}."
            )
        return coremltools.models.MLModel(get_path)

    def save(self, path: PathType) -> None:
        self._model.save(self.get_path(path, self.COREMLMODEL_EXTENSION))
