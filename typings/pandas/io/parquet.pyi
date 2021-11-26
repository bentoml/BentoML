from typing import AnyStr
from pandas import DataFrame
from pandas._typing import FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.util._decorators import doc

def get_engine(engine: str) -> BaseImpl: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame): ...
    def write(self, df: DataFrame, path, compression, **kwargs): ...
    def read(self, path, columns=..., **kwargs): ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePathOrBuffer[AnyStr],
        compression: str | None = ...,
        index: bool | None = ...,
        storage_options: StorageOptions = ...,
        partition_cols: list[str] | None = ...,
        **kwargs
    ): ...
    def read(
        self,
        path,
        columns=...,
        use_nullable_dtypes=...,
        storage_options: StorageOptions = ...,
        **kwargs
    ): ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path,
        compression=...,
        index=...,
        partition_cols=...,
        storage_options: StorageOptions = ...,
        **kwargs
    ): ...
    def read(
        self, path, columns=..., storage_options: StorageOptions = ..., **kwargs
    ): ...

@doc(storage_options=generic._shared_docs["storage_options"])
def to_parquet(
    df: DataFrame,
    path: FilePathOrBuffer | None = ...,
    engine: str = ...,
    compression: str | None = ...,
    index: bool | None = ...,
    storage_options: StorageOptions = ...,
    partition_cols: list[str] | None = ...,
    **kwargs
) -> bytes | None: ...
