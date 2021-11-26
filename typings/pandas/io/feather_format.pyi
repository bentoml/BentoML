from typing import AnyStr
from pandas import DataFrame
from pandas._typing import FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import doc

@doc(storage_options=generic._shared_docs["storage_options"])
def to_feather(
    df: DataFrame,
    path: FilePathOrBuffer[AnyStr],
    storage_options: StorageOptions = ...,
    **kwargs
): ...
