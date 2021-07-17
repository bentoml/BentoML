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

    Class attributes:

    - model (`coremltools.models.MLModel`):
         :class:`~coreml.models.MLModel` instance
    - metadata (`Dict[str, Any]`, `optional`):
        Class metadata

    Raises:
        MissingDependencyException:
            :obj:`coremltools` required by CoreMLModel
        InvalidArgument:
            model is not of instance :class:`~coremltools.models.MLModel`

    Example usage::

        # train.py
        import coremltools as ct
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                ...

            def forward(self, x):
                ...

        net = Net()
        # Train model with data, then convert to CoreML.
        model = ct.convert(net, ...)

        # bento_service.py
        TODO:

        # bento_packer.py
        TODO:
    """

    if int(ct.__version__.split(".")[0]) == 4:
        COREMLMODEL_FILE_EXTENSION = ".mlmodel"
    else:
        # for coremltools>=5.0
        COREMLMODEL_FILE_EXTENSION = ".mlpackage"

    def __init__(
        self,
        model: "ct.models.MLModel",
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        super(CoreMLModel, self).__init__(model, metadata=metadata)
        self.__name__ = 'coremlmodel'

    @classmethod
    def load(cls, path) -> "ct.models.MLModel":
        model_path: Path = cls.ext_path(path, cls.COREMLMODEL_FILE_EXTENSION)
        if not model_path:
            raise InvalidArgument(
                f"given {path} doesn't contain {cls.COREMLMODEL_FILE_EXTENSION} object."
            )
        model = ct.models.MLModel(str(model_path))

        return model

    def save(self, path: PathType) -> None:
        self._model.save(self.model_path(path, self.COREMLMODEL_FILE_EXTENSION))
