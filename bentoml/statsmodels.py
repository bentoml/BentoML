from ._internal.models import PickleModel


class StatsModel(PickleModel):
    """
    Model class saving/loading :obj:`statsmodel` object with pickle serialization
    using ``cloudpickle``. :class:`StatsModel` is a :class:`PickleModel` wrapper.

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """
