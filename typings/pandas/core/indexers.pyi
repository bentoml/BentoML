from typing import TYPE_CHECKING

import numpy as np
from pandas._typing import Any, AnyArrayLike, ArrayLike
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index

"""
Low-dependency indexing utilities.
"""
if TYPE_CHECKING: ...

def is_valid_positional_slice(slc: slice) -> bool:
    """
    Check if a slice object can be interpreted as a positional indexer.

    Parameters
    ----------
    slc : slice

    Returns
    -------
    bool

    Notes
    -----
    A valid positional slice may also be interpreted as a label-based slice
    depending on the index being sliced.
    """
    ...

def is_list_like_indexer(key) -> bool:
    """
    Check if we have a list-like indexer that is *not* a NamedTuple.

    Parameters
    ----------
    key : object

    Returns
    -------
    bool
    """
    ...

def is_scalar_indexer(indexer, ndim: int) -> bool:
    """
    Return True if we are all scalar indexers.

    Parameters
    ----------
    indexer : object
    ndim : int
        Number of dimensions in the object being indexed.

    Returns
    -------
    bool
    """
    ...

def is_empty_indexer(indexer, arr_value: np.ndarray) -> bool:
    """
    Check if we have an empty indexer.

    Parameters
    ----------
    indexer : object
    arr_value : np.ndarray

    Returns
    -------
    bool
    """
    ...

def check_setitem_lengths(indexer, value, values) -> bool:
    """
    Validate that value and indexer are the same length.

    An special-case is allowed for when the indexer is a boolean array
    and the number of true values equals the length of ``value``. In
    this case, no exception is raised.

    Parameters
    ----------
    indexer : sequence
        Key for the setitem.
    value : array-like
        Value for the setitem.
    values : array-like
        Values being set into.

    Returns
    -------
    bool
        Whether this is an empty listlike setting which is a no-op.

    Raises
    ------
    ValueError
        When the indexer is an ndarray or list and the lengths don't match.
    """
    ...

def validate_indices(indices: np.ndarray, n: int) -> None:
    """
    Perform bounds-checking for an indexer.

    -1 is allowed for indicating missing values.

    Parameters
    ----------
    indices : ndarray
    n : int
        Length of the array being indexed.

    Raises
    ------
    ValueError

    Examples
    --------
    >>> validate_indices(np.array([1, 2]), 3) # OK

    >>> validate_indices(np.array([1, -2]), 3)
    Traceback (most recent call last):
        ...
    ValueError: negative dimensions are not allowed

    >>> validate_indices(np.array([1, 2, 3]), 3)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds

    >>> validate_indices(np.array([-1, -1]), 0) # OK

    >>> validate_indices(np.array([0, 1]), 0)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds
    """
    ...

def maybe_convert_indices(indices, n: int, verify: bool = ...):  # -> ndarray:
    """
    Attempt to convert indices into valid, positive indices.

    If we have negative indices, translate to positive here.
    If we have indices that are out-of-bounds, raise an IndexError.

    Parameters
    ----------
    indices : array-like
        Array of indices that we are to convert.
    n : int
        Number of elements in the array that we are indexing.
    verify : bool, default True
        Check that all entries are between 0 and n - 1, inclusive.

    Returns
    -------
    array-like
        An array-like of positive indices that correspond to the ones
        that were passed in initially to this function.

    Raises
    ------
    IndexError
        One of the converted indices either exceeded the number of,
        elements (specified by `n`), or was still negative.
    """
    ...

def is_exact_shape_match(target: ArrayLike, value: ArrayLike) -> bool:
    """
    Is setting this value into this target overwriting the entire column?

    Parameters
    ----------
    target : np.ndarray or ExtensionArray
    value : np.ndarray or ExtensionArray

    Returns
    -------
    bool
    """
    ...

def length_of_indexer(indexer, target=...) -> int:
    """
    Return the expected length of target[indexer]

    Returns
    -------
    int
    """
    ...

def deprecate_ndim_indexing(result, stacklevel: int = ...):  # -> None:
    """
    Helper function to raise the deprecation warning for multi-dimensional
    indexing on 1D Series/Index.

    GH#27125 indexer like idx[:, None] expands dim, but we cannot do that
    and keep an index, so we currently return ndarray, which is deprecated
    (Deprecation GH#30588).
    """
    ...

def unpack_1tuple(tup):
    """
    If we have a length-1 tuple/list that contains a slice, unpack to just
    the slice.

    Notes
    -----
    The list case is deprecated.
    """
    ...

def check_key_length(columns: Index, key, value: DataFrame):  # -> None:
    """
    Checks if a key used as indexer has the same length as the columns it is
    associated with.

    Parameters
    ----------
    columns : Index The columns of the DataFrame to index.
    key : A list-like of keys to index with.
    value : DataFrame The value to set for the keys.

    Raises
    ------
    ValueError: If the length of key is not equal to the number of columns in value
                or if the number of columns referenced by key is not equal to number
                of columns.
    """
    ...

def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
    """
    Check if `indexer` is a valid array indexer for `array`.

    For a boolean mask, `array` and `indexer` are checked to have the same
    length. The dtype is validated, and if it is an integer or boolean
    ExtensionArray, it is checked if there are missing values present, and
    it is converted to the appropriate numpy array. Other dtypes will raise
    an error.

    Non-array indexers (integer, slice, Ellipsis, tuples, ..) are passed
    through as is.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    array : array-like
        The array that is being indexed (only used for the length).
    indexer : array-like or list-like
        The array-like that's used to index. List-like input that is not yet
        a numpy array or an ExtensionArray is converted to one. Other input
        types are passed through as is.

    Returns
    -------
    numpy.ndarray
        The validated indexer as a numpy array that can be used to index.

    Raises
    ------
    IndexError
        When the lengths don't match.
    ValueError
        When `indexer` cannot be converted to a numpy ndarray to index
        (e.g. presence of missing values).

    See Also
    --------
    api.types.is_bool_dtype : Check if `key` is of boolean dtype.

    Examples
    --------
    When checking a boolean mask, a boolean ndarray is returned when the
    arguments are all valid.

    >>> mask = pd.array([True, False])
    >>> arr = pd.array([1, 2])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    An IndexError is raised when the lengths don't match.

    >>> mask = pd.array([True, False, True])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    Traceback (most recent call last):
    ...
    IndexError: Boolean index has wrong length: 3 instead of 2.

    NA values in a boolean array are treated as False.

    >>> mask = pd.array([True, pd.NA])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    A numpy boolean mask will get passed through (if the length is correct):

    >>> mask = np.array([True, False])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    Similarly for integer indexers, an integer ndarray is returned when it is
    a valid indexer, otherwise an error is  (for integer indexers, a matching
    length is not required):

    >>> indexer = pd.array([0, 2], dtype="Int64")
    >>> arr = pd.array([1, 2, 3])
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    array([0, 2])

    >>> indexer = pd.array([0, pd.NA], dtype="Int64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    ValueError: Cannot index with an integer indexer containing NA values

    For non-integer/boolean dtypes, an appropriate error is raised:

    >>> indexer = np.array([0., 2.], dtype="float64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    IndexError: arrays used as indices must be of integer or boolean type
    """
    ...
