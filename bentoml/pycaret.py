from ._internal.artifacts import PickleArtifact


class PycaretModel(PickleArtifact):
    """
    Model class saving/loading :obj:`pycaret` object with pickle serialization
    using ``cloudpickle``. :class:`PycaretModel` is a :class:`PickleArtifact` wrapper.

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """
