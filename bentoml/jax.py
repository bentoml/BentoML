import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException


class FlaxModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`flax` models

    Args:
        model (``):
            TODO:
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

    try:
        import jax
    except ImportError:
        raise MissingDependencyException("jax is required by FlaxModel")

    def __init__(
        self, model, metadata: t.Optional[MetadataType] = None,
    ):
        super(FlaxModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType):
        try:
            import flax
        except ImportError:
            raise MissingDependencyException("flax is required by FlaxModel")
        print(flax.__version__)

    def save(self, path: PathType) -> None:
        try:
            import flax
        except ImportError:
            raise MissingDependencyException("flax is required by FlaxModel")
        print(flax.__version__)


class TraxModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`trax` models

    Args:
        model (``):
            TODO:
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`jax` and :obj:`trax` are required by TraxModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    try:
        import jax
    except ImportError:
        raise MissingDependencyException("jax is required by TraxModel")

    def __init__(
        self, model, metadata: t.Optional[MetadataType] = None,
    ):
        super(TraxModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType):
        try:
            import trax
        except ImportError:
            raise MissingDependencyException("trax is required by FlaxModel")
        print(trax.models.GRULM)

    def save(self, path: PathType) -> None:
        try:
            import trax
        except ImportError:
            raise MissingDependencyException("trax is required by FlaxModel")
        print(trax.models.GRULM)
