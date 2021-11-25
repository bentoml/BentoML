from .logger import Logger

"""
A context object for caching a function's return value each time it
is called with the same input arguments.

"""
FIRST_LINE_TEXT = ...

def extract_first_line(func_code):  # -> tuple[str | Unknown, int]:
    """Extract the first line information from the function code
    text if available.
    """
    ...

class JobLibCollisionWarning(UserWarning):
    """Warn that there might be a collision between names of functions."""

    ...

_STORE_BACKENDS = ...

def register_store_backend(backend_name, backend):  # -> None:
    """Extend available store backends.

    The Memory, MemorizeResult and MemorizeFunc objects are designed to be
    agnostic to the type of store used behind. By default, the local file
    system is used but this function gives the possibility to extend joblib's
    memory pattern with other types of storage such as cloud storage (S3, GCS,
    OpenStack, HadoopFS, etc) or blob DBs.

    Parameters
    ----------
    backend_name: str
        The name identifying the store backend being registered. For example,
        'local' is used with FileSystemStoreBackend.
    backend: StoreBackendBase subclass
        The name of a class that implements the StoreBackendBase interface.

    """
    ...

_FUNCTION_HASHES = ...

class MemorizedResult(Logger):
    """Object representing a cached value.

    Attributes
    ----------
    location: str
        The location of joblib cache. Depends on the store backend used.

    func: function or str
        function whose output is cached. The string case is intended only for
        instanciation based on the output of repr() on another instance.
        (namely eval(repr(memorized_instance)) works).

    argument_hash: str
        hash of the function arguments.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local'.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache numpy arrays. See
        numpy.load for the meaning of the different values.

    verbose: int
        verbosity level (0 means no message).

    timestamp, metadata: string
        for internal use only.
    """

    def __init__(
        self,
        location,
        func,
        args_id,
        backend=...,
        mmap_mode=...,
        verbose=...,
        timestamp=...,
        metadata=...,
    ) -> None: ...
    @property
    def argument_hash(self): ...
    def get(self):  # -> Any:
        """Read value from cache and return it."""
        ...
    def clear(self):  # -> None:
        """Clear value from cache"""
        ...
    def __repr__(self): ...
    def __getstate__(self): ...

class NotMemorizedResult:
    """Class representing an arbitrary value.

    This class is a replacement for MemorizedResult when there is no cache.
    """

    __slots__ = ...
    def __init__(self, value) -> None: ...
    def get(self): ...
    def clear(self): ...
    def __repr__(self): ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...

class NotMemorizedFunc:
    """No-op object decorating a function.

    This class replaces MemorizedFunc when there is no cache. It provides an
    identical API but does not write anything on disk.

    Attributes
    ----------
    func: callable
        Original undecorated function.
    """

    def __init__(self, func) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def call_and_shelve(self, *args, **kwargs): ...
    def __repr__(self): ...
    def clear(self, warn=...): ...
    def call(self, *args, **kwargs): ...
    def check_call_in_cache(self, *args, **kwargs): ...

class MemorizedFunc(Logger):
    """Callable object decorating a function for caching its return value
    each time it is called.

    Methods are provided to inspect the cache or clean it.

    Attributes
    ----------
    func: callable
        The original, undecorated, function.

    location: string
        The location of joblib cache. Depends on the store backend used.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local', in which case the location is the path to a
        disk storage.

    ignore: list or None
        List of variable names to ignore when choosing whether to
        recompute.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache
        numpy arrays. See numpy.load for the meaning of the different
        values.

    compress: boolean, or integer
        Whether to zip the stored data on disk. If an integer is
        given, it should be between 1 and 9, and sets the amount
        of compression. Note that compressed arrays cannot be
        read by memmapping.

    verbose: int, optional
        The verbosity flag, controls messages that are issued as
        the function is evaluated.
    """

    def __init__(
        self,
        func,
        location,
        backend=...,
        ignore=...,
        mmap_mode=...,
        compress=...,
        verbose=...,
        timestamp=...,
    ) -> None: ...
    @property
    def func_code_info(self): ...
    def call_and_shelve(self, *args, **kwargs):  # -> MemorizedResult:
        """Call wrapped function, cache result and return a reference.

        This method returns a reference to the cached result instead of the
        result itself. The reference object is small and pickeable, allowing
        to send or store it easily. Call .get() on reference object to get
        result.

        Returns
        -------
        cached_result: MemorizedResult or NotMemorizedResult
            reference to the value returned by the wrapped function. The
            class "NotMemorizedResult" is used when there is no cache
            activated (e.g. location=None in Memory).
        """
        ...
    def __call__(self, *args, **kwargs): ...
    def __getstate__(self): ...
    def check_call_in_cache(self, *args, **kwargs):
        """Check if function call is in the memory cache.

        Does not call the function or do any work besides func inspection
        and arg hashing.

        Returns
        -------
        is_call_in_cache: bool
            Whether or not the result of the function has been cached
            for the input arguments that have been passed.
        """
        ...
    def clear(self, warn=...):  # -> None:
        """Empty the function's cache."""
        ...
    def call(
        self, *args, **kwargs
    ):  # -> tuple[Unknown, dict[str, float | dict[str | Unknown, str]]]:
        """Force the execution of the function with the given arguments and
        persist the output values.
        """
        ...
    def __repr__(self): ...

class Memory(Logger):
    """A context object for caching a function's return value each time it
    is called with the same input arguments.

    All values are cached on the filesystem, in a deep directory
    structure.

    Read more in the :ref:`User Guide <memory>`.

    Parameters
    ----------
    location: str, pathlib.Path or None
        The path of the base directory to use as a data store
        or None. If None is given, no caching is done and
        the Memory object is completely transparent. This option
        replaces cachedir since version 0.12.

    backend: str, optional
        Type of store backend for reading/writing cache files.
        Default: 'local'.
        The 'local' backend is using regular filesystem operations to
        manipulate data (open, mv, etc) in the backend.

    cachedir: str or None, optional

        .. deprecated: 0.12
            'cachedir' has been deprecated in 0.12 and will be
            removed in 0.14. Use the 'location' parameter instead.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        The memmapping mode used when loading from cache
        numpy arrays. See numpy.load for the meaning of the
        arguments.

    compress: boolean, or integer, optional
        Whether to zip the stored data on disk. If an integer is
        given, it should be between 1 and 9, and sets the amount
        of compression. Note that compressed arrays cannot be
        read by memmapping.

    verbose: int, optional
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated.

    bytes_limit: int, optional
        Limit in bytes of the size of the cache. By default, the size of
        the cache is unlimited. When reducing the size of the cache,
        ``joblib`` keeps the most recently accessed items first.

        **Note:** You need to call :meth:`joblib.Memory.reduce_size` to
        actually reduce the cache size to be less than ``bytes_limit``.

    backend_options: dict, optional
        Contains a dictionnary of named parameters used to configure
        the store backend.
    """

    def __init__(
        self,
        location=...,
        backend=...,
        cachedir=...,
        mmap_mode=...,
        compress=...,
        verbose=...,
        bytes_limit=...,
        backend_options=...,
    ) -> None: ...
    @property
    def cachedir(self): ...
    def cache(
        self, func=..., ignore=..., verbose=..., mmap_mode=...
    ):  # -> partial[Unknown] | NotMemorizedFunc | MemorizedFunc:
        """Decorates the given function func to only compute its return
        value for input arguments not cached on disk.

        Parameters
        ----------
        func: callable, optional
            The function to be decorated
        ignore: list of strings
            A list of arguments name to ignore in the hashing
        verbose: integer, optional
            The verbosity mode of the function. By default that
            of the memory object is used.
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments. By default that of the memory object is used.

        Returns
        -------
        decorated_func: MemorizedFunc object
            The returned object is a MemorizedFunc object, that is
            callable (behaves like a function), but offers extra
            methods for cache lookup and management. See the
            documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        ...
    def clear(self, warn=...):  # -> None:
        """Erase the complete cache directory."""
        ...
    def reduce_size(self):  # -> None:
        """Remove cache elements to make cache size fit in ``bytes_limit``."""
        ...
    def eval(self, func, *args, **kwargs):  # -> Any | None:
        """Eval function func with arguments `*args` and `**kwargs`,
        in the context of the memory.

        This method works similarly to the builtin `apply`, except
        that the function is called only if the cache is not
        up to date.

        """
        ...
    def __repr__(self): ...
    def __getstate__(self):  # -> dict[str, Any]:
        """We don't store the timestamp when pickling, to avoid the hash
        depending from it.
        """
        ...
