from datetime import tzinfo
from typing import TYPE_CHECKING, Any, MutableMapping

import numpy as np
import pyarrow
from pandas import Categorical, Index
from pandas._libs.tslibs import Period, Timestamp, dtypes
from pandas._typing import DtypeObj, NpDtype, Ordered, type_t
from pandas.core.arrays import DatetimeArray, IntervalArray, PandasArray, PeriodArray
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype

"""
Define extension dtypes.
"""
if TYPE_CHECKING: ...
str_type = str

class PandasExtensionDtype(ExtensionDtype):
    """
    A np.dtype duck-typed class, suitable for holding a custom dtype.

    THIS IS NOT A REAL NUMPY DTYPE
    """

    type: Any
    kind: Any
    subdtype = ...
    str: str_type
    num = ...
    shape: tuple[int, ...] = ...
    itemsize = ...
    base: DtypeObj | None = ...
    isbuiltin = ...
    isnative = ...
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = ...
    def __repr__(self) -> str_type:
        """
        Return a string representation for a particular object.
        """
        ...
    def __hash__(self) -> int: ...
    def __getstate__(self) -> dict[str_type, Any]: ...
    @classmethod
    def reset_cache(cls) -> None:
        """clear the cache"""
        ...

class CategoricalDtypeType(type):
    """
    the type of CategoricalDtype, this metaclass determines subclass ability
    """

    ...

@register_extension_dtype
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    """
    Type for categorical data with the categories and orderedness.

    Parameters
    ----------
    categories : sequence, optional
        Must be unique, and must not contain any nulls.
        The categories are stored in an Index,
        and if an index is provided the dtype of that index will be used.
    ordered : bool or None, default False
        Whether or not this categorical is treated as a ordered categorical.
        None can be used to maintain the ordered value of existing categoricals when
        used in operations that combine categoricals, e.g. astype, and will resolve to
        False if there is no existing ordered to maintain.

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    None

    See Also
    --------
    Categorical : Represent a categorical variable in classic R / S-plus fashion.

    Notes
    -----
    This class is useful for specifying the type of a ``Categorical``
    independent of the values. See :ref:`categorical.categoricaldtype`
    for more.

    Examples
    --------
    >>> t = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)
    >>> pd.Series(['a', 'b', 'a', 'c'], dtype=t)
    0      a
    1      b
    2      a
    3    NaN
    dtype: category
    Categories (2, object): ['b' < 'a']

    An empty CategoricalDtype with a specific dtype can be created
    by providing an empty index. As follows,

    >>> pd.CategoricalDtype(pd.DatetimeIndex([])).categories.dtype
    dtype('<M8[ns]')
    """

    name = ...
    type: type[CategoricalDtypeType] = ...
    kind: str_type = ...
    str = ...
    base = ...
    _metadata = ...
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = ...
    def __init__(self, categories=..., ordered: Ordered = ...) -> None: ...
    @classmethod
    def construct_from_string(cls, string: str_type) -> CategoricalDtype:
        """
        Construct a CategoricalDtype from a string.

        Parameters
        ----------
        string : str
            Must be the string "category" in order to be successfully constructed.

        Returns
        -------
        CategoricalDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a CategoricalDtype cannot be constructed from the input.
        """
        ...
    def __setstate__(self, state: MutableMapping[str_type, Any]) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: Any) -> bool:
        """
        Rules for CDT equality:
        1) Any CDT is equal to the string 'category'
        2) Any CDT is equal to itself
        3) Any CDT is equal to a CDT with categories=None regardless of ordered
        4) A CDT with ordered=True is only equal to another CDT with
           ordered=True and identical categories in the same order
        5) A CDT with ordered={False, None} is only equal to another CDT with
           ordered={False, None} and identical categories, but same order is
           not required. There is no distinction between False/None.
        6) Any other comparison returns False
        """
        ...
    def __repr__(self) -> str_type: ...
    @classmethod
    def construct_array_type(cls) -> type_t[Categorical]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...
    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        """
        Validates that we have a valid ordered parameter. If
        it is not a boolean, a TypeError will be raised.

        Parameters
        ----------
        ordered : object
            The parameter to be verified.

        Raises
        ------
        TypeError
            If 'ordered' is not a boolean.
        """
        ...
    @staticmethod
    def validate_categories(categories, fastpath: bool = ...) -> Index:
        """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
        fastpath : bool
            Whether to skip nan and uniqueness checks

        Returns
        -------
        categories : Index
        """
        ...
    def update_dtype(self, dtype: str_type | CategoricalDtype) -> CategoricalDtype:
        """
        Returns a CategoricalDtype with categories and ordered taken from dtype
        if specified, otherwise falling back to self if unspecified

        Parameters
        ----------
        dtype : CategoricalDtype

        Returns
        -------
        new_dtype : CategoricalDtype
        """
        ...
    @property
    def categories(self) -> Index:
        """
        An ``Index`` containing the unique categories allowed.
        """
        ...
    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.
        """
        ...

@register_extension_dtype
class DatetimeTZDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for timezone-aware datetime data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    unit : str, default "ns"
        The precision of the datetime data. Currently limited
        to ``"ns"``.
    tz : str, int, or datetime.tzinfo
        The timezone.

    Attributes
    ----------
    unit
    tz

    Methods
    -------
    None

    Raises
    ------
    pytz.UnknownTimeZoneError
        When the requested timezone cannot be found.

    Examples
    --------
    >>> pd.DatetimeTZDtype(tz='UTC')
    datetime64[ns, UTC]

    >>> pd.DatetimeTZDtype(tz='dateutil/US/Central')
    datetime64[ns, tzfile('/usr/share/zoneinfo/US/Central')]
    """

    type: type[Timestamp] = ...
    kind: str_type = ...
    str = ...
    num = ...
    base = ...
    na_value = ...
    _metadata = ...
    _match = ...
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = ...
    def __init__(self, unit: str_type | DatetimeTZDtype = ..., tz=...) -> None: ...
    @property
    def unit(self) -> str_type:
        """
        The precision of the datetime data.
        """
        ...
    @property
    def tz(self) -> tzinfo:
        """
        The timezone.
        """
        ...
    @classmethod
    def construct_array_type(cls) -> type_t[DatetimeArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...
    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        """
        Construct a DatetimeTZDtype from a string.

        Parameters
        ----------
        string : str
            The string alias for this DatetimeTZDtype.
            Should be formatted like ``datetime64[ns, <tz>]``,
            where ``<tz>`` is the timezone name.

        Examples
        --------
        >>> DatetimeTZDtype.construct_from_string('datetime64[ns, UTC]')
        datetime64[ns, UTC]
        """
        ...
    def __str__(self) -> str_type: ...
    @property
    def name(self) -> str_type:
        """A string representation of the dtype."""
        ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
    def __setstate__(self, state) -> None: ...

@register_extension_dtype
class PeriodDtype(dtypes.PeriodDtypeBase, PandasExtensionDtype):
    """
    An ExtensionDtype for Period data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    freq : str or DateOffset
        The frequency of this PeriodDtype.

    Attributes
    ----------
    freq

    Methods
    -------
    None

    Examples
    --------
    >>> pd.PeriodDtype(freq='D')
    period[D]

    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())
    period[M]
    """

    type: type[Period] = ...
    kind: str_type = ...
    str = ...
    base = ...
    num = ...
    _metadata = ...
    _match = ...
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = ...
    def __new__(cls, freq=...):  # -> PeriodDtype | PandasExtensionDtype:
        """
        Parameters
        ----------
        freq : frequency
        """
        ...
    def __reduce__(self): ...
    @property
    def freq(self):
        """
        The frequency object of this PeriodDtype.
        """
        ...
    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        ...
    def __str__(self) -> str_type: ...
    @property
    def name(self) -> str_type: ...
    @property
    def na_value(self): ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __setstate__(self, state): ...
    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        ...
    @classmethod
    def construct_array_type(cls) -> type_t[PeriodArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...
    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> PeriodArray:
        """
        Construct PeriodArray from pyarrow Array/ChunkedArray.
        """
        ...

@register_extension_dtype
class IntervalDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for Interval data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.

    Attributes
    ----------
    subtype

    Methods
    -------
    None

    Examples
    --------
    >>> pd.IntervalDtype(subtype='int64', closed='both')
    interval[int64, both]
    """

    name = ...
    kind: str_type = ...
    str = ...
    base = ...
    num = ...
    _metadata = ...
    _match = ...
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = ...
    def __new__(cls, subtype=..., closed: str_type | None = ...): ...
    @property
    def closed(self): ...
    @property
    def subtype(self):
        """
        The dtype of the Interval bounds.
        """
        ...
    @classmethod
    def construct_array_type(cls) -> type[IntervalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...
    @classmethod
    def construct_from_string(cls, string: str_type) -> IntervalDtype:
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
        ...
    @property
    def type(self): ...
    def __str__(self) -> str_type: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
    def __setstate__(self, state): ...
    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        ...
    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> IntervalArray:
        """
        Construct IntervalArray from pyarrow Array/ChunkedArray.
        """
        ...

class PandasDtype(ExtensionDtype):
    """
    A Pandas ExtensionDtype for NumPy dtypes.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    dtype : object
        Object to be converted to a NumPy data type object.

    See Also
    --------
    numpy.dtype
    """

    _metadata = ...
    def __init__(self, dtype: NpDtype | PandasDtype | None) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def numpy_dtype(self) -> np.dtype:
        """
        The NumPy dtype this PandasDtype wraps.
        """
        ...
    @property
    def name(self) -> str:
        """
        A bit-width name for this data-type.
        """
        ...
    @property
    def type(self) -> type[np.generic]:
        """
        The type object used to instantiate a scalar of this NumPy data-type.
        """
        ...
    @classmethod
    def construct_from_string(cls, string: str) -> PandasDtype: ...
    @classmethod
    def construct_array_type(cls) -> type_t[PandasArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...
    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV') identifying the general kind of data.
        """
        ...
    @property
    def itemsize(self) -> int:
        """
        The element size of this data-type object.
        """
        ...
