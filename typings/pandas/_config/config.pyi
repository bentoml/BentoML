from contextlib import ContextDecorator, contextmanager
from typing import Any, Callable, Iterable

"""
The config module holds package-wide configurables and provides
a uniform API for working with them.

Overview
========

This module supports the following requirements:
- options are referenced using keys in dot.notation, e.g. "x.y.option - z".
- keys are case-insensitive.
- functions should accept partial/regex keys, when unambiguous.
- options can be registered by modules at import time.
- options can be registered at init-time (via core.config_init)
- options have a default value, and (optionally) a description and
  validation function associated with them.
- options can be deprecated, in which case referencing them
  should produce a warning.
- deprecated options can optionally be rerouted to a replacement
  so that accessing a deprecated option reroutes to a differently
  named option.
- options can be reset to their default value.
- all option can be reset to their default value at once.
- all options in a certain sub - namespace can be reset at once.
- the user can set / get / reset or ask for the description of an option.
- a developer can register and mark an option as deprecated.
- you can register a callback to be invoked when the option value
  is set or reset. Changing the stored value is considered misuse, but
  is not verboten.

Implementation
==============

- Data is stored using nested dictionaries, and should be accessed
  through the provided API.

- "Registered options" and "Deprecated options" have metadata associated
  with them, which are stored in auxiliary dictionaries keyed on the
  fully-qualified key, e.g. "x.y.z.option".

- the config_init module is imported by the package's __init__.py file.
  placing any register_option() calls there will ensure those options
  are available as soon as pandas is loaded. If you use register_option
  in a module, it will only be available after that module is imported,
  which you should be aware of.

- `config_prefix` is a context_manager (for use with the `with` keyword)
  which can save developers some typing, see the docstring.

"""
DeprecatedOption = ...
RegisteredOption = ...
_deprecated_options: dict[str, DeprecatedOption] = ...
_registered_options: dict[str, RegisteredOption] = ...
_global_config: dict[str, Any] = ...
_reserved_keys: list[str] = ...

class OptionError(AttributeError, KeyError):
    """
    Exception for pandas.options, backwards compatible with KeyError
    checks
    """

    ...

def get_default_val(pat: str): ...

class DictWrapper:
    """provide attribute-style access to a nested dict"""

    def __init__(self, d: dict[str, Any], prefix: str = ...) -> None: ...
    def __setattr__(self, key: str, val: Any) -> None: ...
    def __getattr__(self, key: str): ...
    def __dir__(self) -> Iterable[str]: ...

class CallableDynamicDoc:
    def __init__(self, func, doc_tmpl) -> None: ...
    def __call__(self, *args, **kwds): ...
    @property
    def __doc__(self): ...

_get_option_tmpl = ...
_set_option_tmpl = ...
_describe_option_tmpl = ...
_reset_option_tmpl = ...
get_option = ...
set_option = ...
reset_option = ...
describe_option = ...
options = ...

class option_context(ContextDecorator):
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.

    Examples
    --------
    >>> with option_context('display.max_rows', 10, 'display.max_columns', 5):
    ...     ...
    """

    def __init__(self, *args) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args): ...

def register_option(
    key: str,
    defval: object,
    doc: str = ...,
    validator: Callable[[Any], Any] | None = ...,
    cb: Callable[[str], Any] | None = ...,
) -> None:
    """
    Register an option in the package-wide pandas config object

    Parameters
    ----------
    key : str
        Fully-qualified key, e.g. "x.y.option - z".
    defval : object
        Default value of the option.
    doc : str
        Description of the option.
    validator : Callable, optional
        Function of a single argument, should raise `ValueError` if
        called with a value which is not a legal value for the option.
    cb
        a function of a single argument "key", which is called
        immediately after an option value is set/reset. key is
        the full name of the option.

    Raises
    ------
    ValueError if `validator` is specified and `defval` is not a valid value.

    """
    ...

def deprecate_option(
    key: str, msg: str | None = ..., rkey: str | None = ..., removal_ver=...
) -> None:
    """
    Mark option `key` as deprecated, if code attempts to access this option,
    a warning will be produced, using `msg` if given, or a default message
    if not.
    if `rkey` is given, any access to the key will be re-routed to `rkey`.

    Neither the existence of `key` nor that if `rkey` is checked. If they
    do not exist, any subsequence access will fail as usual, after the
    deprecation warning is given.

    Parameters
    ----------
    key : str
        Name of the option to be deprecated.
        must be a fully-qualified option name (e.g "x.y.z.rkey").
    msg : str, optional
        Warning message to output when the key is referenced.
        if no message is given a default message will be emitted.
    rkey : str, optional
        Name of an option to reroute access to.
        If specified, any referenced `key` will be
        re-routed to `rkey` including set/get/reset.
        rkey must be a fully-qualified option name (e.g "x.y.z.rkey").
        used by the default message if no `msg` is specified.
    removal_ver : optional
        Specifies the version in which this option will
        be removed. used by the default message if no `msg` is specified.

    Raises
    ------
    OptionError
        If the specified key has already been deprecated.
    """
    ...

def pp_options_list(keys: Iterable[str], width=..., _print: bool = ...):  # -> str | None:
    """Builds a concise listing of available options, grouped by prefix"""
    ...

@contextmanager
def config_prefix(prefix):  # -> Generator[None, None, None]:
    """
    contextmanager for multiple invocations of API with a common prefix

    supported API functions: (register / get / set )__option

    Warning: This is not thread - safe, and won't work properly if you import
    the API functions into your module using the "from x import y" construct.

    Example
    -------
    import pandas._config.config as cf
    with cf.config_prefix("display.font"):
        cf.register_option("color", "red")
        cf.register_option("size", " 5 pt")
        cf.set_option(size, " 6 pt")
        cf.get_option(size)
        ...

        etc'

    will register options "display.font.color", "display.font.size", set the
    value of "display.font.size"... and so on.
    """
    ...

def is_type_factory(_type: type[Any]) -> Callable[[Any], None]:
    """

    Parameters
    ----------
    `_type` - a type to be compared against (e.g. type(x) == `_type`)

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if type(x) is not equal to `_type`

    """
    ...

def is_instance_factory(_type) -> Callable[[Any], None]:
    """

    Parameters
    ----------
    `_type` - the type to be checked against

    Returns
    -------
    validator - a function of a single argument x , which raises
                ValueError if x is not an instance of `_type`

    """
    ...

def is_one_of_factory(legal_values) -> Callable[[Any], None]: ...
def is_nonnegative_int(value: int | None) -> None:
    """
    Verify that value is None or a positive int.

    Parameters
    ----------
    value : None or int
            The `value` to be checked.

    Raises
    ------
    ValueError
        When the value is not None or is a negative integer
    """
    ...

is_int = ...
is_bool = ...
is_float = ...
is_str = ...
is_text = ...

def is_callable(obj) -> bool:
    """

    Parameters
    ----------
    `obj` - the object to be checked

    Returns
    -------
    validator - returns True if object is callable
        raises ValueError otherwise.

    """
    ...
