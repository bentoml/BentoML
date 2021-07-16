import logging
import os
import re
import typing as t
from pathlib import Path

from bentoml._internal.exceptions import FailedPrecondition, InvalidArgument
from bentoml._internal.utils.ruamel_yaml import YAML

logger = logging.getLogger(__name__)

ARTIFACT_DIR = ".artifacts"


class BaseModelArtifact(object):
    """
    :class:`~_internal.artifacts.BaseModelArtifact` is the base abstraction
    for describing the trained model serialization and deserialization process.

    Class attributes:

    - name (`str`):
        returns name identifier of given Model definition
    """

    def __init__(self, name: str):
        if not name.isidentifier():
            raise ValueError(
                "Artifact name must be a valid python identifier, "
                "a string is considered a valid identifier if it "
                "only contains alphanumeric letters (a-z) and (0-9), "
                "or underscores (_). A valid identifier cannot start "
                "with a number, or contain any spaces."
            )
        self._name = name
        self._metadata = dict()

    @property
    def name(self):
        """
        Identifier of Model class of type :class:`~bentoml._internal.artifacts.BaseModelArtifacts`.

        Returns:
            name of defined Model class.
        """
        return self._name

    @property
    def metadata(self):
        return self._metadata

    def __metadata_path(self, path: str) -> str:
        return os.path.join(path, re.sub("[^-a-zA-Z0-9_.() ]+", "", self.name) + ".yml")

    @classmethod
    def save(
        cls,
        identifier: str,
        model,
        directory: t.Union[str, os.PathLike],
        *,
        metadata: t.Dict[str, t.Union[t.Any, ...]] = None,
        **kw
    ) -> t.Callable:
        inherited = object.__getattribute__(cls, 'save')

        return cls

    @classmethod
    def load(
        cls,
        identifier: str,
        version: t.Optional[int] = None,
        directory: t.Optional[str] = None,
    ) -> t.Type["BaseModelArtifact"]:
        raise NotImplementedError()


def load(identifier: str, directory: str, version: str) -> "BaseModelArtifact":
    ...


def list(path: t.Union[str, os.PathLike]) -> str:
    ...


if __name__ == '__main__':
    t = BaseModelArtifact("my_model")
    print(t)