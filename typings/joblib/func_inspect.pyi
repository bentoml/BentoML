"""
My own variation on function-specific inspect-like features.
"""
full_argspec_fields = ...
full_argspec_type = ...

def get_func_code(
    func,
):  # -> tuple[str, str | Any | Unknown, int] | tuple[str, Unknown, Unknown] | tuple[str, Unknown | str | Any | None, Literal[-1]]:
    """Attempts to retrieve a reliable function code hash.

    The reason we don't use inspect.getsource is that it caches the
    source, whereas we want this to be modified on the fly when the
    function is modified.

    Returns
    -------
    func_code: string
        The function code
    source_file: string
        The path to the file in which the function is defined.
    first_line: int
        The first line of the code in the source file.

    Notes
    ------
    This function does a bit more magic than inspect, and is thus
    more robust.
    """
    ...

def get_func_name(func, resolv_alias=..., win_characters=...):
    """Return the function import path (as a list of module names), and
    a name for the function.

    Parameters
    ----------
    func: callable
        The func to inspect
    resolv_alias: boolean, optional
        If true, possible local aliases are indicated.
    win_characters: boolean, optional
        If true, substitute special characters using urllib.quote
        This is useful in Windows, as it cannot encode some filenames
    """
    ...

def filter_args(func, ignore_lst, args=..., kwargs=...):
    """Filters the given args and kwargs using a list of arguments to
    ignore, and a function specification.

    Parameters
    ----------
    func: callable
        Function giving the argument specification
    ignore_lst: list of strings
        List of arguments to ignore (either a name of an argument
        in the function spec, or '*', or '**')
    *args: list
        Positional arguments passed to the function.
    **kwargs: dict
        Keyword arguments passed to the function

    Returns
    -------
    filtered_args: list
        List of filtered positional and keyword arguments.
    """
    ...

def format_signature(func, *args, **kwargs): ...
def format_call(func, args, kwargs, object_name=...):  # -> str:
    """Returns a nicely formatted statement displaying the function
    call with the given arguments.
    """
    ...
