import contextlib
import pickle as pkl
from typing import TYPE_CHECKING

from pandas import DataFrame, Series

"""
Support pre-0.12 series pickle compatibility.
"""
if TYPE_CHECKING: ...

def load_reduce(self): ...

_sparse_msg = ...

class _LoadSparseSeries:
    def __new__(cls) -> Series: ...

class _LoadSparseFrame:
    def __new__(cls) -> DataFrame: ...

_class_locations_map = ...

class Unpickler(pkl._Unpickler):
    def find_class(self, module, name): ...

def load_newobj(self): ...
def load_newobj_ex(self): ...
def load(fh, encoding: str | None = ..., is_verbose: bool = ...):  # -> Any:
    """
    Load a pickle, with a provided encoding,

    Parameters
    ----------
    fh : a filelike object
    encoding : an optional encoding
    is_verbose : show exception output
    """
    ...

def loads(
    bytes_object: bytes,
    *,
    fix_imports: bool = ...,
    encoding: str = ...,
    errors: str = ...
):  # -> Any:
    """
    Analogous to pickle._loads.
    """
    ...

@contextlib.contextmanager
def patch_pickle():  # -> Generator[None, None, None]:
    """
    Temporarily patch pickle to use our unpickler.
    """
    ...
