import numpy as np
from pandas._typing import ArrayLike
from pandas.core.arrays.numeric import NumericArray, NumericDtype
from pandas.core.dtypes.base import register_extension_dtype
from pandas.util._decorators import cache_readonly

class _IntegerDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size & kind of integer dtype.

    These specific implementations are subclasses of the non-public
    _IntegerDtype. For example we have Int8Dtype to represent signed int 8s.

    The attributes name & type are set when these subclasses are created.
    """

    def __repr__(self) -> str: ...
    @cache_readonly
    def is_signed_integer(self) -> bool: ...
    @cache_readonly
    def is_unsigned_integer(self) -> bool: ...
    @classmethod
    def construct_array_type(cls) -> type[IntegerArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...

def safe_cast(values, dtype, copy: bool):
    """
    Safely cast the values to the dtype if they
    are equivalent, meaning floats must be equivalent to the
    ints.

    """
    ...

def coerce_to_array(
    values, dtype, mask=..., copy: bool = ...
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask

    Parameters
    ----------
    values : 1D list-like
    dtype : integer dtype
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    ...

class IntegerArray(NumericArray):
    """
    Array of integer (optional missing) values.

    .. versionchanged:: 1.0.0

       Now uses :attr:`pandas.NA` as the missing value rather
       than :attr:`numpy.nan`.

    .. warning::

       IntegerArray is currently experimental, and its API or internal
       implementation may change without warning.

    We represent an IntegerArray with 2 numpy arrays:

    - data: contains a numpy integer array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an IntegerArray from generic array-like input, use
    :func:`pandas.array` with one of the integer dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d integer-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    IntegerArray

    Examples
    --------
    Create an IntegerArray with :func:`pandas.array`.

    >>> int_array = pd.array([1, None, 3], dtype=pd.Int32Dtype())
    >>> int_array
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([1, None, 3], dtype='Int32')
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    >>> pd.array([1, None, 3], dtype='UInt16')
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: UInt16
    """

    _internal_fill_value = ...
    @cache_readonly
    def dtype(self) -> _IntegerDtype: ...
    def __init__(
        self, values: np.ndarray, mask: np.ndarray, copy: bool = ...
    ) -> None: ...
    def astype(self, dtype, copy: bool = ...) -> ArrayLike:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        ndarray or ExtensionArray
            NumPy ndarray, BooleanArray or IntegerArray with 'dtype' for its dtype.

        Raises
        ------
        TypeError
            if incompatible type with an IntegerDtype, equivalent of same_kind
            casting
        """
        ...
    def sum(self, *, skipna=..., min_count=..., **kwargs): ...
    def prod(self, *, skipna=..., min_count=..., **kwargs): ...
    def min(self, *, skipna=..., **kwargs): ...
    def max(self, *, skipna=..., **kwargs): ...

_dtype_docstring = ...

@register_extension_dtype
class Int8Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class Int16Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class Int32Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class Int64Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class UInt8Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class UInt16Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class UInt32Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class UInt64Dtype(_IntegerDtype):
    type = ...
    name = ...
    __doc__ = ...

INT_STR_TO_DTYPE: dict[str, _IntegerDtype] = ...
