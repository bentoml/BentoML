import typing as t

from ..types import MetadataType, PathType
from ..utils import cloudpickle
from .base import ModelArtifact


class PickleArtifact(ModelArtifact):
    """
    Abstraction for saving/loading python objects with pickle serialization
    using ``cloudpickle``

    Args:
    model (`Any`, or serializable object):
        Data that can be serialized with :obj:`cloudpickle`
    metadata (`Dict[str, Union[Any,...]]`, `optional`, default to `None`):
        dictionary of model metadata
    name (`str`, `optional`, default to `picklemodel`):
        Name of PickleArtifact instance

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self, model: t.Any, metadata: t.Optional[MetadataType] = None,
    ):
        super(PickleArtifact, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        with open(cls.get_path(path, cls.PICKLE_EXTENSION), 'rb') as inf:
            model = cloudpickle.load(inf)
        return model

    def save(self, path: PathType) -> None:
        with open(self.get_path(path, self.PICKLE_EXTENSION), 'wb') as inf:
            cloudpickle.dump(self._model, inf)
