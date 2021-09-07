import os
import typing as t

import bentoml._internal.constants as _const

from ._internal.models.base import MODEL_NAMESPACE, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="mxnet.gluon",
    module=__name__,
    inst="`pip install mxnet`",
)
if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import mxnet
    import mxnet.gluon as gluon
else:
    mxnet = LazyLoader("mxnet", globals(), "mxnet", exc_msg=_exc)
    gluon = LazyLoader("gluon", globals(), "mxnet.gluon", exc_msg=_exc)


class GluonModel(Model):
    """
    Model class for saving/loading :obj:`mxnet.gluon` models

    Args:
        model (`mxnet.gluon.Block`):
            Every :obj:`mxnet.gluon` object is based on :obj:`mxnet.gluon.Block`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`mxnet` is required by GluonModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    _model: "gluon.HybridBlock"

    def __init__(
        self,
        model: "mxnet.gluon.HybridBlock",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(GluonModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "mxnet.gluon.Block":
        json_path: str = os.path.join(path, f"{MODEL_NAMESPACE}-symbol.json")
        params_path: str = os.path.join(path, f"{MODEL_NAMESPACE}-0000.params")
        return gluon.SymbolBlock.imports(json_path, ["data"], params_path)

    def save(self, path: PathType) -> None:
        self._model.export(os.path.join(path, MODEL_NAMESPACE))
