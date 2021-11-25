import pickle

"""
Fast cryptographic hash of Python objects, with a special case for fast
hashing of numpy arrays.
"""
Pickler = pickle._Pickler

class _ConsistentSet:
    """Class used to ensure the hash of Sets is preserved
    whatever the order of its items.
    """

    def __init__(self, set_sequence) -> None: ...

class _MyHash:
    """Class used to hash objects that won't normally pickle"""

    def __init__(self, *args) -> None: ...

class Hasher(Pickler):
    """A subclass of pickler, to do cryptographic hashing, rather than
    pickling.
    """

    def __init__(self, hash_name=...) -> None: ...
    def hash(self, obj, return_digest=...): ...
    def save(self, obj): ...
    def memoize(self, obj): ...
    def save_global(self, obj, name=..., pack=...): ...
    dispatch = ...
    def save_set(self, set_items): ...

class NumpyHasher(Hasher):
    """Special case the hasher for when numpy is loaded."""

    def __init__(self, hash_name=..., coerce_mmap=...) -> None:
        """
        Parameters
        ----------
        hash_name: string
            The hash algorithm to be used
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
            objects.
        """
        ...
    def save(self, obj):
        """Subclass the save method, to hash ndarray subclass, rather
        than pickling them. Off course, this is a total abuse of
        the Pickler class.
        """
        ...

def hash(obj, hash_name=..., coerce_mmap=...):  # -> str | None:
    """Quick calculation of a hash to identify uniquely Python objects
    containing numpy arrays.


    Parameters
    -----------
    hash_name: 'md5' or 'sha1'
        Hashing algorithm used. sha1 is supposedly safer, but md5 is
        faster.
    coerce_mmap: boolean
        Make no difference between np.memmap and np.ndarray
    """
    ...
