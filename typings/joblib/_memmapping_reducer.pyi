"""
Reducer using memory mapping for numpy arrays
"""
SYSTEM_SHARED_MEM_FS = ...
SYSTEM_SHARED_MEM_FS_MIN_SIZE = ...
FOLDER_PERMISSIONS = ...
FILE_PERMISSIONS = ...
JOBLIB_MMAPS = ...

def add_maybe_unlink_finalizer(memmap): ...
def unlink_file(filename):  # -> None:
    """Wrapper around os.unlink with a retry mechanism.

    The retry mechanism has been implemented primarily to overcome a race
    condition happening during the finalizer of a np.memmap: when a process
    holding the last reference to a mmap-backed np.memmap/np.array is about to
    delete this array (and close the reference), it sends a maybe_unlink
    request to the resource_tracker. This request can be processed faster than
    it takes for the last reference of the memmap to be closed, yielding (on
    Windows) a PermissionError in the resource_tracker loop.
    """
    ...

class _WeakArrayKeyMap:
    """A variant of weakref.WeakKeyDictionary for unhashable numpy arrays.

    This datastructure will be used with numpy arrays as obj keys, therefore we
    do not use the __get__ / __set__ methods to avoid any conflict with the
    numpy fancy indexing syntax.
    """

    def __init__(self) -> None: ...
    def get(self, obj): ...
    def set(self, obj, value): ...
    def __getstate__(self): ...

def has_shareable_memory(a):  # -> bool:
    """Return True if a is backed by some mmap buffer directly or not."""
    ...

def reduce_array_memmap_backward(
    a,
):  # -> tuple[(filename: Unknown, dtype: Unknown, mode: Unknown, offset: Unknown, order: Unknown, shape: Unknown, strides: Unknown, total_buffer_len: Unknown, unlink_on_gc_collect: Unknown) -> Unknown, tuple[Any, Unknown, Any, Any, Literal['F', 'C'], Unknown, Unknown | None, Unknown | None, Literal[False]]] | tuple[(__data: bytes, *, fix_imports: bool = ..., encoding: str = ..., errors: str = ..., buffers: Iterable[Any] | None = ...) -> Any, tuple[bytes]]:
    """reduce a np.array or a np.memmap from a child process"""
    ...

class ArrayMemmapForwardReducer:
    """Reducer callable to dump large arrays to memmap files.

    Parameters
    ----------
    max_nbytes: int
        Threshold to trigger memmapping of large arrays to files created
        a folder.
    temp_folder_resolver: callable
        An callable in charge of resolving a temporary folder name where files
        for backing memmapped arrays are created.
    mmap_mode: 'r', 'r+' or 'c'
        Mode for the created memmap datastructure. See the documentation of
        numpy.memmap for more details. Note: 'w+' is coerced to 'r+'
        automatically to avoid zeroing the data on unpickling.
    verbose: int, optional, 0 by default
        If verbose > 0, memmap creations are logged.
        If verbose > 1, both memmap creations, reuse and array pickling are
        logged.
    prewarm: bool, optional, False by default.
        Force a read on newly memmapped array to make sure that OS pre-cache it
        memory. This can be useful to avoid concurrent disk access when the
        same data array is passed to different worker processes.
    """

    def __init__(
        self,
        max_nbytes,
        temp_folder_resolver,
        mmap_mode,
        unlink_on_gc_collect,
        verbose=...,
        prewarm=...,
    ) -> None: ...
    def __reduce__(self): ...
    def __call__(self, a): ...

def get_memmapping_reducers(
    forward_reducers=...,
    backward_reducers=...,
    temp_folder_resolver=...,
    max_nbytes=...,
    mmap_mode=...,
    verbose=...,
    prewarm=...,
    unlink_on_gc_collect=...,
    **kwargs
):  # -> tuple[dict[Unknown, Unknown] | Unknown, dict[Unknown, Unknown] | Unknown]:
    """Construct a pair of memmapping reducer linked to a tmpdir.

    This function manage the creation and the clean up of the temporary folders
    underlying the memory maps and should be use to get the reducers necessary
    to construct joblib pool or executor.
    """
    ...

class TemporaryResourcesManager:
    """Stateful object able to manage temporary folder and pickles

    It exposes:
    - a per-context folder name resolving API that memmap-based reducers will
      rely on to know where to pickle the temporary memmaps
    - a temporary file/folder management API that internally uses the
      resource_tracker.
    """

    def __init__(self, temp_folder_root=..., context_id=...) -> None: ...
    def set_current_context(self, context_id): ...
    def register_new_context(self, context_id): ...
    def resolve_temp_folder_name(self):
        """Return a folder name specific to the currently activated context"""
        ...
    def register_folder_finalizer(self, pool_subfolder, context_id): ...
