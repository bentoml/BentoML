import numpy as np
from pandas._typing import ArrayLike
from pandas.core.arrays.numeric import NumericArray, NumericDtype
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.util._decorators import cache_readonly

class FloatingDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size of floating dtype.

    These specific implementations are subclasses of the non-public
    FloatingDtype. For example we have Float32Dtype to represent float32.

    The attributes name & type are set when these subclasses are created.
    """

    def __repr__(self) -> str: ...
    @classmethod
    def construct_array_type(cls) -> type[FloatingArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        ...

def coerce_to_array(
    values, dtype=..., mask=..., copy: bool = ...
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    dtype : float dtype
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    ...

class FloatingArray(NumericArray):
    """
    Array of floating (optional missing) values.

    .. versionadded:: 1.2.0

    .. warning::

       FloatingArray is currently experimental, and its API or internal
       implementation may change without warning. Especially the behaviour
       regarding NaN (distinct from NA missing values) is subject to change.

    We represent a FloatingArray with 2 numpy arrays:

    - data: contains a numpy float array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an FloatingArray from generic array-like input, use
    :func:`pandas.array` with one of the float dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d float-dtype array.
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
    FloatingArray

    Examples
    --------
    Create an FloatingArray with :func:`pandas.array`:

    >>> pd.array([0.1, None, 0.3], dtype=pd.Float32Dtype())
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([0.1, None, 0.3], dtype="Float32")
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32
    """

    _internal_fill_value = ...
    @cache_readonly
    def dtype(self) -> FloatingDtype: ...
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
            NumPy ndarray, or BooleanArray, IntegerArray or FloatingArray with
            'dtype' for its dtype.

        Raises
        ------
        TypeError
            if incompatible type with an FloatingDtype, equivalent of same_kind
            casting
        """
        ...
    def sum(self, *, skipna=..., min_count=..., **kwargs): ...
    def prod(self, *, skipna=..., min_count=..., **kwargs): ...
    def min(self, *, skipna=..., **kwargs): ...
    def max(self, *, skipna=..., **kwargs): ...

_dtype_docstring = ...

@register_extension_dtype
class Float32Dtype(FloatingDtype):
    type = ...
    name = ...
    __doc__ = ...

@register_extension_dtype
class Float64Dtype(FloatingDtype):
    type = ...
    name = ...
    __doc__ = ...

FLOAT_STR_TO_DTYPE = ...
