import typing as t
from pathlib import Path

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException
from ._internal.types import PathType

try:
    import coremltools as ct
except ImportError:
    raise MissingDependencyException("coremltools>=4.0b2 is required by CoreMLModel")


class CoreMLModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`coremltools.models.MLModel`
    model that can be used in a BentoML bundle.

    Args:
        model (`coremltools.models.MLModel`):
            :class:`~coreml.models.MLModel` instance
        metadata (`Dict[str, Any]`, `optional`):
            Class metadata
        name (`str`, `optional`, default to `coremlmodel`):
            Optional name for CoreMLModel

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

    if int(ct.__version__.split(".")[0]) == 4:
        COREMLMODEL_FILE_EXTENSION = ".mlmodel"
    else:
        # for coremltools>=5.0
        COREMLMODEL_FILE_EXTENSION = ".mlpackage"
    _model: "ct.models.MLModel"

    def __init__(
        self,
        model: "ct.models.MLModel",
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        name: t.Optional[str] = "coremlmodel",
    ):
        super(CoreMLModel, self).__init__(model, metadata=metadata, name=name)

    @classmethod
    def load(cls, path) -> "ct.models.MLModel":
        model_path: Path = cls.get_path(path, cls.COREMLMODEL_FILE_EXTENSION)
        if not model_path:
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.COREMLMODEL_FILE_EXTENSION}."
            )
        model = ct.models.MLModel(str(model_path))

        return model

    def save(self, path: PathType) -> None:
        self._model.save(self.model_path(path, self.COREMLMODEL_FILE_EXTENSION))
