import logging
import os
import typing as t
from pathlib import Path

import bentoml._internal.constants as _const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import GenericDictType, PathType
from ._internal.utils import LazyLoader

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="spacy",
    module=__name__,
    inst="`pip install spacy`",
)

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import spacy
else:
    spacy = LazyLoader("spacy", globals(), "spacy", exc_msg=_exc)

logger = logging.getLogger(__name__)


class SpacyModel(Model):
    """
    Model class for saving/loading :obj:`spacy` models with `spacy.Language.to_disk` and
    `spacy.util.load_model` methods

    Args:
        model (`spacy.language.Language`):
            Every spacy model is of type :obj:`spacy.language.Language`
        metadata (`GenericDictType`,  `optional`, default to `None`):
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
        metadata: t.Optional[GenericDictType] = None,
    ):
        super(SpacyModel, self).__init__(model, metadata=metadata)

    @classmethod
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

    def save(self, path: PathType) -> None:
        self._model.to_disk(os.path.join(path, MODEL_NAMESPACE))
