import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

try:
    import jax  # noqa # pylint: disable=unused-import
except ImportError:
    raise MissingDependencyException("jax is required by FlaxModel and TraxModel")
try:
    import flax  # pylint: disable=unused-import
except ImportError:
    raise MissingDependencyException("flax is required by FlaxModel")


class FlaxModel(ModelArtifact):
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
        self, model: "flax.linen.Module", metadata: t.Optional[MetadataType] = None,
    ):
        super(FlaxModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "flax.linen.Module":
        ...

    def save(self, path: PathType) -> None:
        ...
