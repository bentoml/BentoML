from typing import Iterable, Sequence

import numpy as np

"""
Module that contains many useful utilities
for validating data or function arguments
"""

def validate_args(fname, args, max_fname_arg_count, compat_args):  # -> None:
    """
    Checks whether the length of the `*args` argument passed into a function
    has at most `len(compat_args)` arguments and whether or not all of these
    elements in `args` are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `*args` parameter
    args : tuple
        The `*args` parameter passed into a function
    max_fname_arg_count : int
        The maximum number of arguments that the function `fname`
        can accept, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args : dict
        A dictionary of keys and their associated default values.
        In order to accommodate buggy behaviour in some versions of `numpy`,
        where a signature displayed keyword arguments but then passed those
        arguments **positionally** internally when calling downstream
        implementations, a dict ensures that the original
        order of the keyword arguments is enforced.

    Raises
    ------
    TypeError
        If `args` contains more values than there are `compat_args`
    ValueError
        If `args` contains values that do not correspond to those
        of the default values specified in `compat_args`
    """
    ...

def validate_kwargs(fname, kwargs, compat_args):  # -> None:
    """
    Checks whether parameters passed to the **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `**kwargs` parameter
    kwargs : dict
        The `**kwargs` parameter passed into `fname`
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to have and their
        associated default values

    Raises
    ------
    TypeError if `kwargs` contains keys not in `compat_args`
    ValueError if `kwargs` contains keys in `compat_args` that do not
    map to the default values specified in `compat_args`
    """
    ...

def validate_args_and_kwargs(
    fname, args, kwargs, max_fname_arg_count, compat_args
):  # -> None:
    """
    Checks whether parameters passed to the *args and **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname: str
        The name of the function being passed the `**kwargs` parameter
    args: tuple
        The `*args` parameter passed into a function
    kwargs: dict
        The `**kwargs` parameter passed into `fname`
    max_fname_arg_count: int
        The minimum number of arguments that the function `fname`
        requires, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to
        have and their associated default values.

    Raises
    ------
    TypeError if `args` contains more values than there are
    `compat_args` OR `kwargs` contains keys not in `compat_args`
    ValueError if `args` contains values not at the default value (`None`)
    `kwargs` contains keys in `compat_args` that do not map to the default
    value as specified in `compat_args`

    See Also
    --------
    validate_args : Purely args validation.
    validate_kwargs : Purely kwargs validation.

    """
    ...

def validate_bool_kwarg(value, arg_name, none_allowed=..., int_allowed=...):  # -> int:
    """
    Ensure that argument passed in arg_name can be interpreted as boolean.

    Parameters
    ----------
    value : bool
        Value to be validated.
    arg_name : str
        Name of the argument. To be reflected in the error message.
    none_allowed : bool, default True
        Whether to consider None to be a valid boolean.
    int_allowed : bool, default False
        Whether to consider integer value to be a valid boolean.

    Returns
    -------
    value
        The same value as input.

    Raises
    ------
    ValueError
        If the value is not a valid boolean.
    """
    ...

def validate_axis_style_args(data, args, kwargs, arg_name, method_name):
    """
    Argument handler for mixed index, columns / axis functions

    In an attempt to handle both `.method(index, columns)`, and
    `.method(arg, axis=.)`, we have to do some bad things to argument
    parsing. This translates all arguments to `{index=., columns=.}` style.

    Parameters
    ----------
    data : DataFrame
    args : tuple
        All positional arguments from the user
    kwargs : dict
        All keyword arguments from the user
    arg_name, method_name : str
        Used for better error messages

    Returns
    -------
    kwargs : dict
        A dictionary of keyword arguments. Doesn't modify ``kwargs``
        inplace, so update them with the return value here.

    Examples
    --------
    >>> df._validate_axis_style_args((str.upper,), {'columns': id},
    ...                              'mapper', 'rename')
    {'columns': <function id>, 'index': <method 'upper' of 'str' objects>}

    This emits a warning
    >>> df._validate_axis_style_args((str.upper, id), {},
    ...                              'mapper', 'rename')
    {'columns': <function id>, 'index': <method 'upper' of 'str' objects>}
    """
    ...

def validate_fillna_kwargs(value, method, validate_scalar_dict_value=...):
    """
    Validate the keyword arguments to 'fillna'.

    This checks that exactly one of 'value' and 'method' is specified.
    If 'method' is specified, this validates that it's a valid method.

    Parameters
    ----------
    value, method : object
        The 'value' and 'method' keyword arguments for 'fillna'.
    validate_scalar_dict_value : bool, default True
        Whether to validate that 'value' is a scalar or dict. Specifically,
        validate that it is not a list or tuple.

    Returns
    -------
    value, method : object
    """
    ...

def validate_percentile(q: float | Iterable[float]) -> np.ndarray:
    """
    Validate percentiles (used by describe and quantile).

    This function checks if the given float or iterable of floats is a valid percentile
    otherwise raises a ValueError.

    Parameters
    ----------
    q: float or iterable of floats
        A single percentile or an iterable of percentiles.

    Returns
    -------
    ndarray
        An ndarray of the percentiles if valid.

    Raises
    ------
    ValueError if percentiles are not in given interval([0, 1]).
    """
    ...

def validate_ascending(
    ascending: bool | int | Sequence[bool | int] = ...,
):  # -> bool | int | Sequence[bool | int] | list[bool | int]:
    """Validate ``ascending`` kwargs for ``sort_index`` method."""
    ...
