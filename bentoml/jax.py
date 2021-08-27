import typing as t

from ._internal.models.base import Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

if t.TYPE_CHECKING:
    import flax  # pylint: disable=unused-import
    import jax  # pylint: disable=unused-import
else:
    jax = LazyLoader("jax", globals(), "jax")
    flax = LazyLoader("flax", globals(), "flax")


class FlaxModel(Model):
    """
    Model class for saving/loading :obj:`flax` models

    Args:
        model (`flax.linen.Module`):
            Every Flax model is of type :obj:`flax.linen.Module`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`jax` and :obj:`flax` are required by FlaxModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    def __init__(
        self,
        model: "flax.linen.Module",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(FlaxModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "flax.linen.Module":
        ...

    def save(self, path: PathType) -> None:
        ...
