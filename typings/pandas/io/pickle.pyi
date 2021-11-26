from typing import Any
from pandas._typing import CompressionOptions, FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import doc

@doc(storage_options=generic._shared_docs["storage_options"])
def to_pickle(
    obj: Any,
    filepath_or_buffer: FilePathOrBuffer,
    compression: CompressionOptions = ...,
    protocol: int = ...,
    storage_options: StorageOptions = ...,
): ...
