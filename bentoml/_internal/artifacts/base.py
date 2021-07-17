import logging
import os
import typing as t
from pathlib import Path

from ..exceptions import InvalidArgument
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
        _instance = super(BaseArtifact, cls).__new__(cls)
        _instance.__init__(*args, **kwargs)
        return _instance

    def __init__(
        self, model: MT, metadata: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        if any(isinstance(model, s) for s in [str, int]):
            raise InvalidArgument(f"cannot accept type {type(model)} as model type")

        self._model = model
        self._metadata = metadata

    @property
    def metadata(self):
        return self._metadata

    def _meta_path(self, path: PathType) -> PathType:
        return PathType(os.path.join(path, f"{self._model.__name__}.yml"))

    def save(self, path: PathType):
        """
        Perform save instance to given path.
        """

    @classmethod
    def load(cls, path: PathType) -> MT:
        inherited = object.__getattribute__(cls, "load")
        return inherited(path)

    def __getattribute__(self, item):
        if item == 'save':

            def wrapped_save(*args, **kw):
                # avoid method overriding
                path = args[0]  # save(self, path)
                if self.metadata:
                    yaml = YAML()
                    yaml.dump(self.metadata, Path(self._meta_path(path)))

                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_save

        return object.__getattribute__(self, item)
