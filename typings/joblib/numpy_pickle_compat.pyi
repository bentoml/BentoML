from .numpy_pickle_utils import Unpickler

"""Numpy pickle compatibility functions."""

def hex_str(an_int):  # -> str:
    """Convert an int to an hexadecimal string."""
    ...

def asbytes(s): ...

_MAX_LEN = ...
_CHUNK_SIZE = ...

def read_zfile(file_handle):  # -> bytes:
    """Read the z-file and return the content as a string.

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guaranteed. Do not
    use for external purposes.
    """
    ...

def write_zfile(file_handle, data, compress=...):  # -> None:
    """Write the data in the given file as a Z-file.

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guarantied. Do not
    use for external purposes.
    """
    ...

class NDArrayWrapper:
    """An object to be persisted instead of numpy arrays.

    The only thing this object does, is to carry the filename in which
    the array has been persisted, and the array subclass.
    """

    def __init__(self, filename, subclass, allow_mmap=...) -> None:
        """Constructor. Store the useful information for later."""
        ...
    def read(self, unpickler):
        """Reconstruct the array."""
        ...

class ZNDArrayWrapper(NDArrayWrapper):
    """An object to be persisted instead of numpy arrays.

    This object store the Zfile filename in which
    the data array has been persisted, and the meta information to
    retrieve it.
    The reason that we store the raw buffer data of the array and
    the meta information, rather than array representation routine
    (tobytes) is that it enables us to use completely the strided
    model to avoid memory copies (a and a.T store as fast). In
    addition saving the heavy information separately can avoid
    creating large temporary buffers when unpickling data with
    large arrays.
    """

    def __init__(self, filename, init_args, state) -> None:
        """Constructor. Store the useful information for later."""
        ...
    def read(self, unpickler):
        """Reconstruct the array from the meta-information and the z-file."""
        ...

class ZipNumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles."""

    dispatch = ...
    def __init__(self, filename, file_handle, mmap_mode=...) -> None:
        """Constructor."""
        ...
    def load_build(self):  # -> None:
        """Set the state of a newly created object.

        We capture it to replace our place-holder objects,
        NDArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        """
        ...

def load_compatibility(filename):  # -> Any:
    """Reconstruct a Python object from a file persisted with joblib.dump.

    This function ensures the compatibility with joblib old persistence format
    (<= 0.9.3).

    Parameters
    -----------
    filename: string
        The name of the file from which to load the object

    Returns
    -------
    result: any Python object
        The object stored in the file.

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump.
    """
    ...
