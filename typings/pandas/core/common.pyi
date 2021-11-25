import contextlib
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterable, Iterator

import numpy as np
from pandas import Index
from pandas._typing import AnyArrayLike, NpDtype, Scalar, T

"""
Misc tools for implementing data structures

Note: pandas.core.common is *not* part of the public API.
"""
if TYPE_CHECKING: ...

class SettingWithCopyError(ValueError): ...
class SettingWithCopyWarning(Warning): ...

def flatten(line):  # -> Generator[Unknown, None, None]:
    """
    Flatten an arbitrarily nested sequence.

    Parameters
    ----------
    line : sequence
        The non string sequence to flatten

    Notes
    -----
    This doesn't consider strings sequences.

    Returns
    -------
    flattened : generator
    """
    ...

def consensus_name_attr(objs): ...
def is_bool_indexer(key: Any) -> bool:
    """
    Check whether `key` is a valid boolean indexer.

    Parameters
    ----------
    key : Any
        Only list-likes may be considered boolean indexers.
        All other types are not considered a boolean indexer.
        For array-like input, boolean ndarrays or ExtensionArrays
        with ``_is_boolean`` set are considered boolean indexers.

    Returns
    -------
    bool
        Whether `key` is a valid boolean indexer.

    Raises
    ------
    ValueError
        When the array is an object-dtype ndarray or ExtensionArray
        and contains missing values.

    See Also
    --------
    check_array_indexer : Check that `key` is a valid array to index,
        and convert to an ndarray.
    """
    ...

def cast_scalar_indexer(val, warn_float: bool = ...):  # -> int:
    """
    To avoid numpy DeprecationWarnings, cast float to integer where valid.

    Parameters
    ----------
    val : scalar
    warn_float : bool, default False
        If True, issue deprecation warning for a float indexer.

    Returns
    -------
    outval : scalar
    """
    ...

def not_none(*args):  # -> Generator[Unknown, None, None]:
    """
    Returns a generator consisting of the arguments that are not None.
    """
    ...

def any_none(*args) -> bool:
    """
    Returns a boolean indicating if any argument is None.
    """
    ...

def all_none(*args) -> bool:
    """
    Returns a boolean indicating if all arguments are None.
    """
    ...

def any_not_none(*args) -> bool:
    """
    Returns a boolean indicating if any argument is not None.
    """
    ...

def all_not_none(*args) -> bool:
    """
    Returns a boolean indicating if all arguments are not None.
    """
    ...

def count_not_none(*args) -> int:
    """
    Returns the count of arguments that are not None.
    """
    ...

def asarray_tuplesafe(values, dtype: NpDtype | None = ...) -> np.ndarray: ...
def index_labels_to_array(labels, dtype: NpDtype | None = ...) -> np.ndarray:
    """
    Transform label or iterable of labels to array, for use in Index.

    Parameters
    ----------
    dtype : dtype
        If specified, use as dtype of the resulting array, otherwise infer.

    Returns
    -------
    array
    """
    ...

def maybe_make_list(obj): ...
def maybe_iterable_to_list(obj: Iterable[T] | T) -> Collection[T] | T:
    """
    If obj is Iterable but not list-like, consume into list.
    """
    ...

def is_null_slice(obj) -> bool:
    """
    We have a null slice.
    """
    ...

def is_true_slices(line) -> list[bool]:
    """
    Find non-trivial slices in "line": return a list of booleans with same length.
    """
    ...

def is_full_slice(obj, line: int) -> bool:
    """
    We have a full length slice.
    """
    ...

def get_callable_name(obj): ...
def apply_if_callable(maybe_callable, obj, **kwargs):
    """
    Evaluate possibly callable input using obj and kwargs if it is callable,
    otherwise return as it is.

    Parameters
    ----------
    maybe_callable : possibly a callable
    obj : NDFrame
    **kwargs
    """
    ...

def standardize_mapping(
    into,
):  # -> partial[defaultdict[Unknown, Unknown]] | Type[Mapping[Unknown, Unknown]]:
    """
    Helper function to standardize a supplied mapping.

    Parameters
    ----------
    into : instance or subclass of collections.abc.Mapping
        Must be a class, an initialized collections.defaultdict,
        or an instance of a collections.abc.Mapping subclass.

    Returns
    -------
    mapping : a collections.abc.Mapping subclass or other constructor
        a callable object that can accept an iterator to create
        the desired Mapping.

    See Also
    --------
    DataFrame.to_dict
    Series.to_dict
    """
    ...

def random_state(state=...):  # -> RandomState | Module("numpy.random"):
    """
    Helper function for processing random_state arguments.

    Parameters
    ----------
    state : int, array-like, BitGenerator (NumPy>=1.17), np.random.RandomState, None.
        If receives an int, array-like, or BitGenerator, passes to
        np.random.RandomState() as seed.
        If receives an np.random.RandomState object, just returns object.
        If receives `None`, returns np.random.
        If receives anything else, raises an informative ValueError.

        .. versionchanged:: 1.1.0

            array-like and BitGenerator (for NumPy>=1.18) object now passed to
            np.random.RandomState() as seed

        Default None.

    Returns
    -------
    np.random.RandomState or np.random if state is None

    """
    ...

def pipe(
    obj, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
) -> T:
    """
    Apply a function ``func`` to object ``obj`` either by passing obj as the
    first argument to the function or, in the case that the func is a tuple,
    interpret the first element of the tuple as a function and pass the obj to
    that function as a keyword argument whose key is the value of the second
    element of the tuple.

    Parameters
    ----------
    func : callable or tuple of (callable, str)
        Function to apply to this object or, alternatively, a
        ``(callable, data_keyword)`` tuple where ``data_keyword`` is a
        string indicating the keyword of `callable`` that expects the
        object.
    *args : iterable, optional
        Positional arguments passed into ``func``.
    **kwargs : dict, optional
        A dictionary of keyword arguments passed into ``func``.

    Returns
    -------
    object : the return type of ``func``.
    """
    ...

def get_rename_function(mapper):  # -> (x: Unknown) -> Unknown:
    """
    Returns a function that will map names/labels, dependent if mapper
    is a dict, Series or just a function.
    """
    ...

def convert_to_list_like(values: Scalar | Iterable | AnyArrayLike) -> list | AnyArrayLike:
    """
    Convert list-like or scalar input to list-like. List, numpy and pandas array-like
    inputs are returned unmodified whereas others are converted to list.
    """
    ...

@contextlib.contextmanager
def temp_setattr(obj, attr: str, value) -> Iterator[None]:
    """Temporarily set attribute on an object.

    Args:
        obj: Object whose attribute will be modified.
        attr: Attribute to modify.
        value: Value to temporarily set attribute to.

    Yields:
        obj with modified attribute.
    """
    ...

def require_length_match(data, index: Index):  # -> None:
    """
    Check the length of data matches the length of the index.
    """
    ...

_builtin_table = ...
_cython_table = ...

def get_cython_func(arg: Callable) -> str | None:
    """
    if we define an internal function for this argument, return it
    """
    ...

def is_builtin_func(
    arg,
):  # -> (a: tuple[_RecursiveSequence | Unknown], axis: _ShapeLike = ..., dtype: DTypeLike = ..., out: ndarray | None = ..., keepdims: bool = ..., initial: _NumberLike_co = ..., where: tuple[Unknown] = ...) -> Any | (a: Unknown, axis: Unknown = ..., out: Unknown = ..., keepdims: Unknown = ..., initial: Unknown = ..., where: Unknown = ...) -> Unknown:
    """
    if we define an builtin function for this argument, return it,
    otherwise return the arg
    """
    ...
