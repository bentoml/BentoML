from contextlib import contextmanager
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    Hashable,
    Iterable,
    Sequence,
)
import numpy as np
from pandas import DataFrame, Series
from pandas._typing import (
    ColspaceArgType,
    CompressionOptions,
    FilePathOrBuffer,
    FloatFormatType,
    FormattersType,
    IndexLabel,
    StorageOptions,
)
from pandas.core.arrays import Categorical, DatetimeArray, TimedeltaArray
from pandas.core.indexes.api import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING: ...
common_docstring = ...
_VALID_JUSTIFY_PARAMETERS = ...
return_docstring = ...

class CategoricalFormatter:
    def __init__(
        self,
        categorical: Categorical,
        buf: IO[str] | None = ...,
        length: bool = ...,
        na_rep: str = ...,
        footer: bool = ...,
    ) -> None: ...
    def to_string(self) -> str: ...

class SeriesFormatter:
    def __init__(
        self,
        series: Series,
        buf: IO[str] | None = ...,
        length: bool | str = ...,
        header: bool = ...,
        index: bool = ...,
        na_rep: str = ...,
        name: bool = ...,
        float_format: str | None = ...,
        dtype: bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> None: ...
    def to_string(self) -> str: ...

class TextAdjustment:
    def __init__(self) -> None: ...
    def len(self, text: str) -> int: ...
    def justify(self, texts: Any, max_len: int, mode: str = ...) -> list[str]: ...
    def adjoin(self, space: int, *lists, **kwargs) -> str: ...

class EastAsianTextAdjustment(TextAdjustment):
    def __init__(self) -> None: ...
    def len(self, text: str) -> int: ...
    def justify(
        self, texts: Iterable[str], max_len: int, mode: str = ...
    ) -> list[str]: ...

def get_adjustment() -> TextAdjustment: ...

class DataFrameFormatter:
    __doc__ = ...
    def __init__(
        self,
        frame: DataFrame,
        columns: Sequence[str] | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: bool | Sequence[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        justify: str | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool | str = ...,
        decimal: str = ...,
        bold_rows: bool = ...,
        escape: bool = ...,
    ) -> None: ...
    def get_strcols(self) -> list[list[str]]: ...
    @property
    def should_show_dimensions(self) -> bool: ...
    @property
    def is_truncated(self) -> bool: ...
    @property
    def is_truncated_horizontally(self) -> bool: ...
    @property
    def is_truncated_vertically(self) -> bool: ...
    @property
    def dimensions_info(self) -> str: ...
    @property
    def has_index_names(self) -> bool: ...
    @property
    def has_column_names(self) -> bool: ...
    @property
    def show_row_idx_names(self) -> bool: ...
    @property
    def show_col_idx_names(self) -> bool: ...
    @property
    def max_rows_displayed(self) -> int: ...
    def truncate(self) -> None: ...
    def format_col(self, i: int) -> list[str]: ...

class DataFrameRenderer:
    def __init__(self, fmt: DataFrameFormatter) -> None: ...
    def to_latex(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        column_format: str | None = ...,
        longtable: bool = ...,
        encoding: str | None = ...,
        multicolumn: bool = ...,
        multicolumn_format: str | None = ...,
        multirow: bool = ...,
        caption: str | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> str | None: ...
    def to_html(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        encoding: str | None = ...,
        classes: str | list | tuple | None = ...,
        notebook: bool = ...,
        border: int | None = ...,
        table_id: str | None = ...,
        render_links: bool = ...,
    ) -> str | None: ...
    def to_string(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        encoding: str | None = ...,
        line_width: int | None = ...,
    ) -> str | None: ...
    def to_csv(
        self,
        path_or_buf: FilePathOrBuffer[AnyStr] | None = ...,
        encoding: str | None = ...,
        sep: str = ...,
        columns: Sequence[Hashable] | None = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        quotechar: str = ...,
        line_terminator: str | None = ...,
        chunksize: int | None = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        errors: str = ...,
        storage_options: StorageOptions = ...,
    ) -> str | None: ...

def save_to_buffer(
    string: str, buf: FilePathOrBuffer[str] | None = ..., encoding: str | None = ...
) -> str | None: ...
@contextmanager
def get_buffer(buf: FilePathOrBuffer[str] | None, encoding: str | None = ...): ...
def format_array(
    values: Any,
    formatter: Callable | None,
    float_format: FloatFormatType | None = ...,
    na_rep: str = ...,
    digits: int | None = ...,
    space: str | int | None = ...,
    justify: str = ...,
    decimal: str = ...,
    leading_space: bool | None = ...,
    quoting: int | None = ...,
) -> list[str]: ...

class GenericArrayFormatter:
    def __init__(
        self,
        values: Any,
        digits: int = ...,
        formatter: Callable | None = ...,
        na_rep: str = ...,
        space: str | int = ...,
        float_format: FloatFormatType | None = ...,
        justify: str = ...,
        decimal: str = ...,
        quoting: int | None = ...,
        fixed_width: bool = ...,
        leading_space: bool | None = ...,
    ) -> None: ...
    def get_result(self) -> list[str]: ...

class FloatArrayFormatter(GenericArrayFormatter):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_result_as_array(self) -> np.ndarray: ...

class IntArrayFormatter(GenericArrayFormatter): ...

class Datetime64Formatter(GenericArrayFormatter):
    def __init__(
        self,
        values: np.ndarray | Series | DatetimeIndex | DatetimeArray,
        nat_rep: str = ...,
        date_format: None = ...,
        **kwargs
    ) -> None: ...

class ExtensionArrayFormatter(GenericArrayFormatter): ...

def format_percentiles(
    percentiles: (np.ndarray | list[int | float] | list[float] | list[str | float]),
) -> list[str]: ...
def is_dates_only(
    values: np.ndarray | DatetimeArray | Index | DatetimeIndex,
) -> bool: ...
def get_format_datetime64(
    is_dates_only: bool, nat_rep: str = ..., date_format: str | None = ...
) -> Callable: ...
def get_format_datetime64_from_values(
    values: np.ndarray | DatetimeArray | DatetimeIndex, date_format: str | None
) -> str | None: ...

class Datetime64TZFormatter(Datetime64Formatter): ...

class Timedelta64Formatter(GenericArrayFormatter):
    def __init__(
        self,
        values: np.ndarray | TimedeltaIndex,
        nat_rep: str = ...,
        box: bool = ...,
        **kwargs
    ) -> None: ...

def get_format_timedelta64(
    values: np.ndarray | TimedeltaIndex | TimedeltaArray,
    nat_rep: str = ...,
    box: bool = ...,
) -> Callable: ...

class EngFormatter:
    ENG_PREFIXES = ...
    def __init__(
        self, accuracy: int | None = ..., use_eng_prefix: bool = ...
    ) -> None: ...
    def __call__(self, num: int | float) -> str: ...

def set_eng_float_format(accuracy: int = ..., use_eng_prefix: bool = ...) -> None: ...
def get_level_lengths(
    levels: Any, sentinel: bool | object | str = ...
) -> list[dict[int, int]]: ...
def buffer_put_lines(buf: IO[str], lines: list[str]) -> None: ...
