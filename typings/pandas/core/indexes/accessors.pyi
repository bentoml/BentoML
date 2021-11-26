from typing import TYPE_CHECKING
import numpy as np
from pandas import Series
from pandas.core.accessor import PandasDelegate, delegate_names
from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
from pandas.core.base import NoNewAttributesMixin, PandasObject

if TYPE_CHECKING: ...

class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs = ...
    def __init__(self, data: Series, orig) -> None: ...

@delegate_names(
    delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_methods, typ="method"
)
class DatetimeProperties(Properties):
    def to_pydatetime(self) -> np.ndarray: ...
    @property
    def freq(self): ...
    def isocalendar(self): ...
    @property
    def weekofyear(self): ...
    week = ...

@delegate_names(
    delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=TimedeltaArray,
    accessors=TimedeltaArray._datetimelike_methods,
    typ="method",
)
class TimedeltaProperties(Properties):
    def to_pytimedelta(self) -> np.ndarray: ...
    @property
    def components(self): ...
    @property
    def freq(self): ...

@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ="method"
)
class PeriodProperties(Properties): ...

class CombinedDatetimelikeProperties(
    DatetimeProperties, TimedeltaProperties, PeriodProperties
):
    def __new__(cls, data: Series): ...
