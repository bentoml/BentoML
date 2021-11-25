from pandas.compat.chainmap import DeepChainMap

"""
Module for scope operations
"""

def ensure_scope(
    level: int, global_dict=..., local_dict=..., resolvers=..., target=..., **kwargs
) -> Scope:
    """Ensure that we are grabbing the correct scope."""
    ...

DEFAULT_GLOBALS = ...

class Scope:
    """
    Object to hold scope, with a few bells to deal with some custom syntax
    and contexts added by pandas.

    Parameters
    ----------
    level : int
    global_dict : dict or None, optional, default None
    local_dict : dict or Scope or None, optional, default None
    resolvers : list-like or None, optional, default None
    target : object

    Attributes
    ----------
    level : int
    scope : DeepChainMap
    target : object
    temps : dict
    """

    __slots__ = ...
    level: int
    scope: DeepChainMap
    resolvers: DeepChainMap
    temps: dict
    def __init__(
        self, level: int, global_dict=..., local_dict=..., resolvers=..., target=...
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def has_resolvers(self) -> bool:
        """
        Return whether we have any extra scope.

        For example, DataFrames pass Their columns as resolvers during calls to
        ``DataFrame.eval()`` and ``DataFrame.query()``.

        Returns
        -------
        hr : bool
        """
        ...
    def resolve(self, key: str, is_local: bool):
        """
        Resolve a variable name in a possibly local context.

        Parameters
        ----------
        key : str
            A variable name
        is_local : bool
            Flag indicating whether the variable is local or not (prefixed with
            the '@' symbol)

        Returns
        -------
        value : object
            The value of a particular variable
        """
        ...
    def swapkey(self, old_key: str, new_key: str, new_value=...) -> None:
        """
        Replace a variable name, with a potentially new value.

        Parameters
        ----------
        old_key : str
            Current variable name to replace
        new_key : str
            New variable name to replace `old_key` with
        new_value : object
            Value to be replaced along with the possible renaming
        """
        ...
    def add_tmp(self, value) -> str:
        """
        Add a temporary variable to the scope.

        Parameters
        ----------
        value : object
            An arbitrary object to be assigned to a temporary variable.

        Returns
        -------
        str
            The name of the temporary variable created.
        """
        ...
    @property
    def ntemps(self) -> int:
        """The number of temporary variables in this scope"""
        ...
    @property
    def full_scope(self) -> DeepChainMap:
        """
        Return the full scope for use with passing to engines transparently
        as a mapping.

        Returns
        -------
        vars : DeepChainMap
            All variables in this scope.
        """
        ...
