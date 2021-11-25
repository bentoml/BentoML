from pandas.util._decorators import doc

"""

accessor.py contains base classes for implementing accessor properties
that can be mixed into or pinned onto other pandas classes.

"""

class DirNamesMixin:
    _accessors: set[str] = ...
    _hidden_attrs: frozenset[str] = ...
    def __dir__(self) -> list[str]:
        """
        Provide method name lookup and completion.

        Notes
        -----
        Only provide 'public' methods.
        """
        ...

class PandasDelegate:
    """
    Abstract base class for delegating methods/properties.
    """

    ...

def delegate_names(
    delegate, accessors, typ: str, overwrite: bool = ...
):  # -> (cls: Unknown) -> Unknown:
    """
    Add delegated names to a class using a class decorator.  This provides
    an alternative usage to directly calling `_add_delegate_accessors`
    below a class definition.

    Parameters
    ----------
    delegate : object
        The class to get methods/properties & doc-strings.
    accessors : Sequence[str]
        List of accessor to add.
    typ : {'property', 'method'}
    overwrite : bool, default False
       Overwrite the method/property in the target class if it exists.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    @delegate_names(Categorical, ["categories", "ordered"], "property")
    class CategoricalAccessor(PandasDelegate):
        [...]
    """
    ...

class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : cls
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.
    """

    def __init__(self, name: str, accessor) -> None: ...
    def __get__(self, obj, cls): ...

@doc(_register_accessor, klass="DataFrame")
def register_dataframe_accessor(name): ...
@doc(_register_accessor, klass="Series")
def register_series_accessor(name): ...
@doc(_register_accessor, klass="Index")
def register_index_accessor(name): ...
