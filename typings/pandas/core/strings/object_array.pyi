from pandas.core.strings.base import BaseStringArrayMethods

class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """

    _str_na_value = ...
    def __len__(self): ...
