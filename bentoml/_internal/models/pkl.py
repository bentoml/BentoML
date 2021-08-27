import os
import typing as t

import cloudpickle

from ..types import MetadataType, PathType
from .base import MODEL_NAMESPACE, PICKLE_EXTENSION, Model


class PickleModel(Model):
    """
    Abstraction for saving/loading python objects with pickle serialization
    using ``cloudpickle``

    Args:
        model (`Any`, or serializable object):
            Data that can be serialized with :obj:`cloudpickle`
        metadata (`Dict[str, Union[Any,...]]`, `optional`, default to `None`):
            dictionary of model metadata

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(
        self,
        model: t.Any,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(PickleModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> t.Any:
        with open(
            os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}"), "rb"
        ) as inf:
            model = cloudpickle.load(inf)
        return model

    def save(self, path: PathType) -> None:
        with open(
            os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}"), "wb"
        ) as inf:
            cloudpickle.dump(self._model, inf)
