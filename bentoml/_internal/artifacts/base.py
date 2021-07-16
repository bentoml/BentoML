import logging
import os
import typing as t
from pathlib import Path

from ..types import MT, PathType
from ..utils.ruamel_yaml import YAML

logger = logging.getLogger(__name__)


class BaseArtifact:
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

    def __new__(cls, *args, **kwargs):
        return super(BaseArtifact, cls).__new__(cls, *args, **kwargs)

    def __init__(
        self, model: MT, metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        self._model = model
        if metadata:
            self._metadata = metadata

    @property
    def metadata(self):
        return self._metadata

    @property
    def model(self):
        return self._model

    def __metadata_path(self, path: PathType) -> PathType:
        # NOTE: will self._model always an object type?
        return PathType(os.path.join(path, f"{self._model.__name__}.yml"))

    def save(self, path: PathType) -> t.Callable:
        if self.metadata:
            yaml = YAML()
            yaml.dump(self.metadata, Path(self.__metadata_path(path)))

        inherited = object.__getattribute__(self, "save")
        return inherited(path)

    @classmethod
    def load(cls, path: PathType) -> MT:
        inherited = object.__getattribute__(cls, "load")
        return inherited(path)
