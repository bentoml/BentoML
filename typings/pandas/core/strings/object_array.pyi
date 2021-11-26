from pandas.core.strings.base import BaseStringArrayMethods

class ObjectStringArrayMixin(BaseStringArrayMethods):
    _str_na_value = ...
