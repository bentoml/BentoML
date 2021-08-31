import logging
import os
import typing as t
from pathlib import Path

import bentoml._internal.constants as const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader, catch_exceptions
from .exceptions import MissingDependencyException

_exc = MissingDependencyException(
    const.IMPORT_ERROR_MSG.format(
        fwr="spacy",
        module=__name__,
        inst="`pip install spacy`",
    )
)

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import spacy
else:
    spacy = LazyLoader("spacy", globals(), "spacy")

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
        self,
        model: "spacy.language.Language",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(SpacyModel, self).__init__(model, metadata=metadata)

    @classmethod
    @catch_exceptions(catch_exc=ModuleNotFoundError, throw_exc=_exc)
    def load(cls, path: PathType, **load_model_kwargs) -> "spacy.language.Language":
        if Path(path).exists():
            name = os.path.join(path, MODEL_NAMESPACE)
            model = spacy.util.load_model(name, **load_model_kwargs)
        else:
            name = path
            if name.startswith("blank:"):  # shortcut for blank model
                model = spacy.util.get_lang_class(name.replace("blank:", ""))
            else:
                try:  # then we try loading model from the package
                    spacy.util.load_model_from_package(name, **load_model_kwargs)
                except (ModuleNotFoundError, OSError):
                    logger.warning(
                        f"module {name} is not available. Installing from pip..."
                    )
                    spacy.cli.download(name)
                finally:
                    model = spacy.util.load_model(name, **load_model_kwargs)
        return model

    @catch_exceptions(catch_exc=ModuleNotFoundError, throw_exc=_exc)
    def save(self, path: PathType) -> None:
        self._model.to_disk(os.path.join(path, MODEL_NAMESPACE))
