import os
import typing as t

import bentoml._internal.constants as const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

_exc = const.IMPORT_ERROR_MSG.format(
    fwr="fasttext",
    module=__name__,
    inst="`pip install fasttext`",
)
if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import fasttext
else:
    fasttext = LazyLoader("fasttext", globals(), "fasttext", exc_msg=_exc)


class FastTextModel(Model):
    """
    Model class for saving/loading :obj:`fasttext` models

    Args:
        model (`fasttext.FastText._FastText`):
            Base pipeline for all fasttext model
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`fasttext` is required by FastTextModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    _model: "fasttext.FastText._FastText"

    def __init__(
        self,
        model: "fasttext.FastText._FastText",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(FastTextModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "fasttext.FastText._FastText":
        return fasttext.load_model(os.path.join(path, MODEL_NAMESPACE))

    def save(self, path: PathType) -> None:
        self._model.save_model(os.path.join(path, MODEL_NAMESPACE))
