import os
import typing as t

import bentoml._internal.constants as _const

from ._internal.models.base import Model
from ._internal.types import GenericDictType, PathType
from ._internal.utils import LazyLoader

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="h2o",
    module=__name__,
    inst="Refers to"
    " https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html#install-in-python",  # noqa
)

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import h2o
    import h2o.model as hm
else:
    h2o = LazyLoader("h2o", globals(), "h2o", exc_msg=_exc)
    hm = LazyLoader("hm", globals(), "h2o.model", exc_msg=_exc)


class H2OModel(Model):
    """
    Model class for saving/loading :obj:`h2o` models
     using meth:`~h2o.saved_model` and :meth:`~h2o.load_model`

    Args:
        model (`h2o.model.model_base.ModelBase`):
            :obj:`ModelBase` for all h2o model instance
        metadata (`GenericDictType`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`h2o` is required by H2OModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(
        self,
        model: "hm.model_base.ModelBase",
        metadata: t.Optional[GenericDictType] = None,
    ):
        super(H2OModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "hm.model_base.ModelBase":
        h2o.init()
        h2o.no_progress()
        # NOTE: model_path should be the first item in
        #   h2o saved artifact directory
        model_path: str = str(os.path.join(path, os.listdir(path)[0]))
        return h2o.load_model(model_path)

    def save(self, path: PathType) -> None:
        h2o.save_model(model=self._model, path=str(path), force=True)
