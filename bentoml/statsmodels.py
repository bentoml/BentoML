from ._internal.artifacts import PickleArtifact


class StatsModel(PickleArtifact):
    """
    Model class saving/loading :obj:`statsmodel` object with pickle serialization
    using ``cloudpickle``. :class:`StatsModel` is a :class:`PickleArtifact` wrapper.

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """
