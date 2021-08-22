import logging
import os
import typing as t
from pathlib import Path

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import spacy
except ImportError:
    raise MissingDependencyException("spacy is required by SpacyModel")

logger = logging.getLogger(__name__)


class SpacyModel(Model):
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

    One then can define :code:`bento.py`::

        TODO:

    """

    def __init__(
        self, model: spacy.language.Language, metadata: t.Optional[MetadataType] = None
    ):
        super(SpacyModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType, **load_model_kwargs) -> "spacy.language.Language":
        if Path(path).exists():
            name = os.path.join(path, MODEL_NAMESPACE)
        else:
            name = path
            if name.startswith("blank:"):  # shortcut for blank model
                return spacy.util.get_lang_class(name.replace("blank:", ""))
            else:
                try:  # then we try loading model from the package
                    spacy.util.load_model_from_package(name, **load_model_kwargs)
                except OSError:
                    spacy.cli.download(name)
        return spacy.util.load_model(name, **load_model_kwargs)

    def save(self, path: PathType) -> None:
        self._model.to_disk(os.path.join(path, MODEL_NAMESPACE))
