import logging
import os
import re
import typing as t
from pathlib import Path

from ..utils.ruamel_yaml import YAML

logger = logging.getLogger(__name__)

MT = t.TypeVar('MT')


class ArtifactMeta(type):
    """Metaclass for treating artifacts subclass as singleton"""

    _singleton: t.Dict["ArtifactMeta", str] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._singleton:
            cls._singleton[cls] = super(ArtifactMeta, cls).__call__(*args, **kwargs)
        return cls._singleton[cls]


class BaseModelArtifact(metaclass=ArtifactMeta):
    """
    :class:`~bentoml._internal.artifacts.BaseModelArtifact` is the base abstraction
    for describing the trained model serialization and deserialization process.
    :class:`~bentoml._internal.artifacts.BaseModelArtifact` is a singleton

    Class attributes:

    - model (`torch.nn.Module`, `tf.keras.Model`, `sklearn.svm.SVC` and many more):
        returns name identifier of given Model definition
    - metadata (`Dict[str, Union[Any,...]]`):
        dictionary of model metadata

    """

    def __init__(
        self, model: MT, metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        self._model = model
        if metadata:
            self._metadata = metadata

    @property
    def metadata(self):
        return self._metadata

    @staticmethod
    def __metadata_path(path: str) -> t.Union[str, os.PathLike]:
        return os.path.join(
            path, re.sub("[^-a-zA-Z0-9_.() ]+", "", "metadata") + ".yml"
        )

    def save(self, path: t.Union[str, os.PathLike]) -> t.Callable:
        if self.metadata:
            yaml = YAML()
            yaml.dump(self.metadata, Path(self.__metadata_path(path)))

        inherited = object.__getattribute__(self, 'save')
        return inherited(path)

    @classmethod
    def load(cls, path: t.Union[str, os.PathLike]) -> MT:
        inherited = object.__getattribute__(cls, 'load')
        return inherited(path)