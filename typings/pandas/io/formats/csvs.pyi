from typing import TYPE_CHECKING, Hashable, Sequence
import numpy as np
from pandas._typing import (
    CompressionOptions,
    FilePathOrBuffer,
    FloatFormatType,
    IndexLabel,
    StorageOptions,
)
from pandas.core.indexes.api import Index
from pandas.io.formats.format import DataFrameFormatter

if TYPE_CHECKING: ...

class CSVFormatter:
    cols: np.ndarray
    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: FilePathOrBuffer[str] | FilePathOrBuffer[bytes] = ...,
        sep: str = ...,
        cols: Sequence[Hashable] | None = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        errors: str = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        line_terminator=...,
        chunksize: int | None = ...,
        quotechar: str | None = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @property
    def na_rep(self) -> str: ...
    @property
    def float_format(self) -> FloatFormatType | None: ...
    @property
    def decimal(self) -> str: ...
    @property
    def header(self) -> bool | Sequence[str]: ...
    @property
    def index(self) -> bool: ...
    @property
    def has_mi_columns(self) -> bool: ...
    @property
    def data_index(self) -> Index: ...
    @property
    def nlevels(self) -> int: ...
    @property
    def write_cols(self) -> Sequence[Hashable]: ...
    @property
    def encoded_labels(self) -> list[Hashable]: ...
    def save(self) -> None: ...
