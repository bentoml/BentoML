from ._internal.models import PickleModel


class PycaretModel(PickleModel):
    """
    Model class saving/loading :obj:`pycaret` object with pickle serialization
    using ``cloudpickle``. :class:`PycaretModel` is a :class:`PickleModel` wrapper.

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """
