
from typing import Generic, Union

__all__ = ["Store", "StoreKey"]
T = ...
D = ...
class StoreKey(Generic[T]):
    """StoreKey is an object used as a key to a Store.

    A StoreKey is associated with the type T of the value of the key.

    A StoreKey is unique and cannot conflict with another key.
    """
    __slots__ = ...


class Store:
    """Store is a type-safe heterogenous mutable mapping that
    allows keys and value types to be defined separately from
    where it (the Store) is created.

    Usually you will be given an object which has a ``Store``:

    .. code-block:: python

        store: Store = some_object.store

    If a module wants to store data in this Store, it creates StoreKeys
    for its keys (at the module level):

    .. code-block:: python

        some_str_key = StoreKey[str]()
        some_bool_key = StoreKey[bool]()

    To store information:

    .. code-block:: python

        # Value type must match the key.
        store[some_str_key] = "value"
        store[some_bool_key] = True

    To retrieve the information:

    .. code-block:: python

        # The static type of some_str is str.
        some_str = store[some_str_key]
        # The static type of some_bool is bool.
        some_bool = store[some_bool_key]

    Why use this?
    -------------

    Problem: module Internal defines an object. Module External, which
    module Internal doesn't know about, receives the object and wants to
    attach information to it, to be retrieved later given the object.

    Bad solution 1: Module External assigns private attributes directly on
    the object. This doesn't work well because the type checker doesn't
    know about these attributes and it complains about undefined attributes.

    Bad solution 2: module Internal adds a ``Dict[str, Any]`` attribute to
    the object. Module External stores its data in private keys of this dict.
    This doesn't work well because retrieved values are untyped.

    Good solution: module Internal adds a ``Store`` to the object. Module
    External mints StoreKeys for its own keys. Module External stores and
    retrieves its data using these keys.
    """
    __slots__ = ...
    def __init__(self) -> None:
        ...
    
    def __setitem__(self, key: StoreKey[T], value: T) -> None:
        """Set a value for key."""
        ...
    
    def __getitem__(self, key: StoreKey[T]) -> T:
        """Get the value for key.

        Raises ``KeyError`` if the key wasn't set before.
        """
        ...
    
    def get(self, key: StoreKey[T], default: D) -> Union[T, D]:
        """Get the value for key, or return default if the key wasn't set
        before."""
        ...
    
    def setdefault(self, key: StoreKey[T], default: T) -> T:
        """Return the value of key if already set, otherwise set the value
        of key to default and return default."""
        ...
    
    def __delitem__(self, key: StoreKey[T]) -> None:
        """Delete the value for key.

        Raises ``KeyError`` if the key wasn't set before.
        """
        ...
    
    def __contains__(self, key: StoreKey[T]) -> bool:
        """Return whether key was set."""
        ...
    


