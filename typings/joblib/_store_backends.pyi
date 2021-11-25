from abc import ABCMeta, abstractmethod

"""Storage providers backends for Memory caching."""
CacheItemInfo = ...

def concurrency_safe_write(object_to_write, filename, write_func):  # -> str:
    """Writes an object into a unique file in a concurrency-safe way."""
    ...

class StoreBackendBase(metaclass=ABCMeta):
    """Helper Abstract Base Class which defines all methods that
    a StorageBackend must implement."""

    location = ...
    @abstractmethod
    def create_location(self, location):  # -> None:
        """Creates a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory.
        """
        ...
    @abstractmethod
    def clear_location(self, location):  # -> None:
        """Clears a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory or a filename absolute path
        """
        ...
    @abstractmethod
    def get_items(self):  # -> None:
        """Returns the whole list of items available in the store.

        Returns
        -------
        The list of items identified by their ids (e.g filename in a
        filesystem).
        """
        ...
    @abstractmethod
    def configure(self, location, verbose=..., backend_options=...):  # -> None:
        """Configures the store.

        Parameters
        ----------
        location: string
            The base location used by the store. On a filesystem, this
            corresponds to a directory.
        verbose: int
            The level of verbosity of the store
        backend_options: dict
            Contains a dictionnary of named paremeters used to configure the
            store backend.
        """
        ...

class StoreBackendMixin:
    """Class providing all logic for managing the store in a generic way.

    The StoreBackend subclass has to implement 3 methods: create_location,
    clear_location and configure. The StoreBackend also has to provide
    a private _open_item, _item_exists and _move_item methods. The _open_item
    method has to have the same signature as the builtin open and return a
    file-like object.
    """

    def load_item(self, path, verbose=..., msg=...):  # -> Any:
        """Load an item from the store given its path as a list of
        strings."""
        ...
    def dump_item(self, path, item, verbose=...):
        """Dump an item in the store at the path given as a list of
        strings."""
        ...
    def clear_item(self, path):  # -> None:
        """Clear the item at the path, given as a list of strings."""
        ...
    def contains_item(self, path):
        """Check if there is an item at the path, given as a list of
        strings"""
        ...
    def get_item_info(self, path):  # -> dict[str, str]:
        """Return information about item."""
        ...
    def get_metadata(self, path):  # -> Any | dict[Unknown, Unknown]:
        """Return actual metadata of an item."""
        ...
    def store_metadata(self, path, metadata):  # -> None:
        """Store metadata of a computation."""
        ...
    def contains_path(self, path):
        """Check cached function is available in store."""
        ...
    def clear_path(self, path):  # -> None:
        """Clear all items with a common path in the store."""
        ...
    def store_cached_func_code(self, path, func_code=...):  # -> None:
        """Store the code of the cached function."""
        ...
    def get_cached_func_code(self, path):
        """Store the code of the cached function."""
        ...
    def get_cached_func_info(self, path):  # -> dict[str, str]:
        """Return information related to the cached function if it exists."""
        ...
    def clear(self):  # -> None:
        """Clear the whole store content."""
        ...
    def reduce_store_size(self, bytes_limit):
        """Reduce store size to keep it under the given bytes limit."""
        ...
    def __repr__(self):  # -> str:
        """Printable representation of the store location."""
        ...

class FileSystemStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""

    _open_item = ...
    _item_exists = ...
    _move_item = ...
    def clear_location(self, location):  # -> None:
        """Delete location on store."""
        ...
    def create_location(self, location):  # -> None:
        """Create object location on store"""
        ...
    def get_items(self):  # -> list[Unknown]:
        """Returns the whole list of items available in the store."""
        ...
    def configure(self, location, verbose=..., backend_options=...):  # -> None:
        """Configure the store backend.

        For this backend, valid store options are 'compress' and 'mmap_mode'
        """
        ...
