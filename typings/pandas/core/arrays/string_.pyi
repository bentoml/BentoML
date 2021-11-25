from typing import TYPE_CHECKING, Any

import pyarrow
from pandas._typing import Scalar, type_t
from pandas.core.arrays import PandasArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype

if TYPE_CHECKING: ...

@register_extension_dtype
class StringDtype(ExtensionDtype):
    """
    Extension dtype for string data.

    .. versionadded:: 1.0.0

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

       In particular, StringDtype.na_value may change to no longer be
       ``numpy.nan``.

    Parameters
    ----------
    storage : {"python", "pyarrow"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.StringDtype()
    string[python]

    >>> pd.StringDtype(storage="pyarrow")
    string[pyarrow]
    """

    name = ...
    na_value = ...
    _metadata = ...
    def __init__(self, storage=...) -> None: ...
    @property
    def type(self) -> type[str]: ...
    @classmethod
    def construct_from_string(cls, string):  # -> Self@StringDtype:
        """
        Construct a StringDtype from a string.

        Parameters
        ----------
        string : str
            The type of the name. The storage type will be taking from `string`.
            Valid options and their storage types are

            ========================== ==============================================
            string                     result storage
            ========================== ==============================================
            ``'string'``               pd.options.mode.string_storage, default python
            ``'string[python]'``       python
            ``'string[pyarrow]'``      pyarrow
            ========================== ==============================================

        Returns
        -------
        StringDtype

        Raise
        -----
        TypeError
            If the string is not a valid option.

        """
        ...
    def __eq__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def construct_array_type(self) -> type_t[BaseStringArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...
    def __repr__(self): ...
    def __str__(self) -> str: ...
    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseStringArray:
        """
        Construct StringArray from pyarrow Array/ChunkedArray.
        """
        ...

class BaseStringArray(ExtensionArray): ...

class StringArray(BaseStringArray, PandasArray):
    """
    Extension array for string data.

    .. versionadded:: 1.0.0

    .. warning::

       StringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : array-like
        The array of data.

        .. warning::

           Currently, this expects an object-dtype ndarray
           where the elements are Python strings or :attr:`pandas.NA`.
           This may change without warning in the future. Use
           :meth:`pandas.array` with ``dtype="string"`` for a stable way of
           creating a `StringArray` from any sequence.

    copy : bool, default False
        Whether to copy the array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    array
        The recommended function for creating a StringArray.
    Series.str
        The string methods are available on Series backed by
        a StringArray.

    Notes
    -----
    StringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string")
    <StringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string

    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``
    will convert the values to strings.

    >>> pd.array(['1', 1], dtype="object")
    <PandasArray>
    ['1', 1]
    Length: 2, dtype: object
    >>> pd.array(['1', 1], dtype="string")
    <StringArray>
    ['1', '1']
    Length: 2, dtype: string

    However, instantiating StringArrays directly with non-strings will raise an error.

    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:

    >>> pd.array(["a", None, "c"], dtype="string") == "a"
    <BooleanArray>
    [True, <NA>, False]
    Length: 3, dtype: boolean
    """

    _typ = ...
    def __init__(self, values, copy=...) -> None: ...
    def __arrow_array__(self, type=...):
        """
        Convert myself into a pyarrow Array.
        """
        ...
    def __setitem__(self, key, value): ...
    def astype(self, dtype, copy=...): ...
    def min(self, axis=..., skipna: bool = ..., **kwargs) -> Scalar: ...
    def max(self, axis=..., skipna: bool = ..., **kwargs) -> Scalar: ...
    def value_counts(self, dropna: bool = ...): ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    _arith_method = ...
    _str_na_value = ...
