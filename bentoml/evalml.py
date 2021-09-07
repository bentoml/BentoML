import os
import typing as t

import bentoml._internal.constants as _const

from ._internal.models.base import MODEL_NAMESPACE, PICKLE_EXTENSION, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="evalml",
    module=__name__,
    inst="`pip install evalml --no-dependencies`",
)

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import evalml
    import evalml.pipelines as pipelines
else:
    evalml = LazyLoader("evalml", globals(), "evalml", exc_msg=_exc)
    pipelines = LazyLoader("pipelines", globals(), "evalml.pipelines", exc_msg=_exc)


class EvalMLModel(Model):
    """
    Model class for saving/loading :obj:`evalml` models

    Args:
        model (`evalml.pipelines.PipelineBase`):
            Base pipeline for all EvalML model
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`evalml` is required by EvalMLModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    _model: "pipelines.PipelineBase"

    def __init__(
        self,
        model: "pipelines.PipelineBase",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(EvalMLModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "pipelines.PipelineBase":
        model_file_path: str = os.path.join(
            path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}"
        )
        return pipelines.PipelineBase.load(model_file_path)

    def save(self, path: PathType) -> None:
        self._model.save(os.path.join(path, f"{MODEL_NAMESPACE}{PICKLE_EXTENSION}"))
