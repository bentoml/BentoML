import logging
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import spacy
except ImportError:
    raise MissingDependencyException("spacy is required by SpacyModel")

logger = logging.getLogger(__name__)


class SpacyModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`spacy` models with `spacy.Language.to_disk` and
    `spacy.util.load_model` methods

    Args:
        model (`spacy.language.Language`):
            Every spacy model is of type :obj:`spacy.language.Language`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`spacy` is required by SpacyModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self, model: spacy.language.Language, metadata: t.Optional[MetadataType] = None
    ):
        super(SpacyModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "spacy.language.Language":
        return spacy.util.load_model(cls.get_path(path))

    def save(self, path: PathType) -> None:
        self._model.to_disk(self.get_path(path))
