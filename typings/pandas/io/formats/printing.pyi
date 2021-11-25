from typing import Any, Callable, Dict, Iterable, Mapping, Sized, Union

"""
Printing tools.
"""
EscapeChars = Union[Mapping[str, str], Iterable[str]]
_KT = ...
_VT = ...

def adjoin(space: int, *lists: list[str], **kwargs) -> str:
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    ----------
    space : int
        number of spaces for padding
    lists : str
        list of str which being joined
    strlen : callable
        function used to calculate the length of each str. Needed for unicode
        handling.
    justfunc : callable
        function used to justify str. Needed for unicode handling.
    """
    ...

def justify(texts: Iterable[str], max_len: int, mode: str = ...) -> list[str]:
    """
    Perform ljust, center, rjust against string or list-like
    """
    ...

def pprint_thing(
    thing: Any,
    _nest_lvl: int = ...,
    escape_chars: EscapeChars | None = ...,
    default_escapes: bool = ...,
    quote_strings: bool = ...,
    max_seq_items: int | None = ...,
) -> str:
    """
    This function is the sanctioned way of converting objects
    to a string representation and properly handles nested sequences.

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list or dict, optional
        Characters to escape. If a dict is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : int or None, default None
        Pass through to other pretty printers to limit sequence printing

    Returns
    -------
    str
    """
    ...

def pprint_thing_encoded(object, encoding: str = ..., errors: str = ...) -> bytes: ...
def enable_data_resource_formatter(enable: bool) -> None: ...
def default_pprint(thing: Any, max_seq_items: int | None = ...) -> str: ...
def format_object_summary(
    obj,
    formatter: Callable,
    is_justify: bool = ...,
    name: str | None = ...,
    indent_for_name: bool = ...,
    line_break_each_value: bool = ...,
) -> str:
    """
    Return the formatted obj as a unicode string

    Parameters
    ----------
    obj : object
        must be iterable and support __getitem__
    formatter : callable
        string formatter for an element
    is_justify : bool
        should justify the display
    name : name, optional
        defaults to the class name of the obj
    indent_for_name : bool, default True
        Whether subsequent lines should be indented to
        align with the name.
    line_break_each_value : bool, default False
        If True, inserts a line break for each value of ``obj``.
        If False, only break lines when the a line of values gets wider
        than the display width.

        .. versionadded:: 0.25.0

    Returns
    -------
    summary string
    """
    ...

def format_object_attrs(
    obj: Sized, include_dtype: bool = ...
) -> list[tuple[str, str | int]]:
    """
    Return a list of tuples of the (attr, formatted_value)
    for common attrs, including dtype, name, length

    Parameters
    ----------
    obj : object
        Must be sized.
    include_dtype : bool
        If False, dtype won't be in the returned list

    Returns
    -------
    list of 2-tuple

    """
    ...

class PrettyDict(Dict[_KT, _VT]):
    """Dict extension to support abbreviated __repr__"""

    def __repr__(self) -> str: ...
