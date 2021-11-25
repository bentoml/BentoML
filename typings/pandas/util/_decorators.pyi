from typing import Any, Callable, Mapping

from pandas._typing import F

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = ...,
    klass: type[Warning] | None = ...,
    stacklevel: int = ...,
    msg: str | None = ...,
) -> Callable[[F], F]:
    """
    Return a new function that emits a deprecation warning on use.

    To use this method for a deprecated function, another function
    `alternative` with the same signature must exist. The deprecated
    function will emit a deprecation warning, and in the docstring
    it will contain the deprecation directive with the provided version
    so it can be detected for future removal.

    Parameters
    ----------
    name : str
        Name of function to deprecate.
    alternative : func
        Function to use instead.
    version : str
        Version of pandas in which the method has been deprecated.
    alt_name : str, optional
        Name to use in preference of alternative.__name__.
    klass : Warning, default FutureWarning
    stacklevel : int, default 2
    msg : str
        The message to display in the warning.
        Default is '{name} is deprecated. Use {alt_name} instead.'
    """
    ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = ...,
    stacklevel: int = ...,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a keyword argument of a function.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.

    Examples
    --------
    The following deprecates 'cols', using 'columns' instead

    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name='columns')
    ... def f(columns=''):
    ...     print(columns)
    ...
    >>> f(columns='should work ok')
    should work ok

    >>> f(cols='should raise warning')
    FutureWarning: cols is deprecated, use columns instead
      warnings.warn(msg, FutureWarning)
    should raise warning

    >>> f(cols='should error', columns="can\'t pass do both")
    TypeError: Can only specify 'cols' or 'columns', not both

    >>> @deprecate_kwarg('old', 'new', {'yes': True, 'no': False})
    ... def f(new=False):
    ...     print('yes!' if new else 'no!')
    ...
    >>> f(old='yes')
    FutureWarning: old='yes' is deprecated, use new=True instead
      warnings.warn(msg, FutureWarning)
    yes!

    To raise a warning that a keyword will be removed entirely in the future

    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name=None)
    ... def f(cols='', another_param=''):
    ...     print(cols)
    ...
    >>> f(cols='should raise warning')
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    >>> f(another_param='should not raise warning')
    should not raise warning

    >>> f(cols='should raise warning', another_param='')
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    """
    ...

def future_version_msg(version: str | None) -> str:
    """Specify which version of pandas the deprecation will take place in."""
    ...

def deprecate_nonkeyword_arguments(
    version: str | None, allowed_args: list[str] | None = ..., stacklevel: int = ...
) -> Callable[[F], F]:
    """
    Decorator to deprecate a use of non-keyword arguments of a function.

    Parameters
    ----------
    version : str, optional
        The version in which positional arguments will become
        keyword-only. If None, then the warning message won't
        specify any particular version.

    allowed_args : list, optional
        In case of list, it must be the list of names of some
        first arguments of the decorated functions that are
        OK to be given as positional arguments. In case of None value,
        defaults to list of all arguments not having the
        default value.

    stacklevel : int, default=2
        The stack level for warnings.warn
    """
    ...

def rewrite_axis_style_signature(
    name: str, extra_params: list[tuple[str, Any]]
) -> Callable[..., Any]: ...
def doc(*docstrings: str | Callable, **params) -> Callable[[F], F]:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : str or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """
    ...

class Substitution:
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """

    def __init__(self, *args, **kwargs) -> None: ...
    def __call__(self, func: F) -> F: ...
    def update(self, *args, **kwargs) -> None:
        """
        Update self.params with supplied args.
        """
        ...

class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    addendum: str | None
    def __init__(
        self, addendum: str | None, join: str = ..., indents: int = ...
    ) -> None: ...
    def __call__(self, func: F) -> F: ...

def indent(text: str | None, indents: int = ...) -> str: ...
