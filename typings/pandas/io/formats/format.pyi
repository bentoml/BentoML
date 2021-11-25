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

"""
Internal module for formatting output data in csv, html, xml,
and latex files. This module also applies to display formatting.
"""
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
    def len(self, text: str) -> int:
        """
        Calculate display width considering unicode East Asian Width
        """
        ...
    def justify(
        self, texts: Iterable[str], max_len: int, mode: str = ...
    ) -> list[str]: ...

def get_adjustment() -> TextAdjustment: ...

class DataFrameFormatter:
    """Class for processing dataframe formatting options and data."""

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
    def get_strcols(self) -> list[list[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        ...
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
    def truncate(self) -> None:
        """
        Check whether the frame should be truncated. If so, slice the frame up.
        """
        ...
    def format_col(self, i: int) -> list[str]: ...

class DataFrameRenderer:
    """Class for creating dataframe output in multiple formats.

    Called in pandas.core.generic.NDFrame:
        - to_csv
        - to_latex

    Called in pandas.core.frame.DataFrame:
        - to_html
        - to_string

    Parameters
    ----------
    fmt : DataFrameFormatter
        Formatter with the formatting options.
    """

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
    ) -> str | None:
        """
        Render a DataFrame to a LaTeX tabular/longtable environment output.
        """
        ...
    def to_html(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        encoding: str | None = ...,
        classes: str | list | tuple | None = ...,
        notebook: bool = ...,
        border: int | None = ...,
        table_id: str | None = ...,
        render_links: bool = ...,
    ) -> str | None:
        """
        Render a DataFrame to a html table.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        classes : str or list-like
            classes to include in the `class` attribute of the opening
            ``<table>`` tag, in addition to the default "dataframe".
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            ``<table>`` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        """
        ...
    def to_string(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        encoding: str | None = ...,
        line_width: int | None = ...,
    ) -> str | None:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        encoding: str, default “utf-8”
            Set character encoding.
        line_width : int, optional
            Width to wrap a line in characters.
        """
        ...
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
    ) -> str | None:
        """
        Render dataframe as comma-separated file.
        """
        ...

def save_to_buffer(
    string: str, buf: FilePathOrBuffer[str] | None = ..., encoding: str | None = ...
) -> str | None:
    """
    Perform serialization. Write to buf or return as string if buf is None.
    """
    ...

@contextmanager
def get_buffer(
    buf: FilePathOrBuffer[str] | None, encoding: str | None = ...
):  # -> Generator[str | IO[str] | RawIOBase | BufferedIOBase | TextIOBase | TextIOWrapper | mmap | StringIO, None, None]:
    """
    Context manager to open, yield and close buffer for filenames or Path-like
    objects, otherwise yield buf unchanged.
    """
    ...

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
) -> list[str]:
    """
    Format an array for printing.

    Parameters
    ----------
    values
    formatter
    float_format
    na_rep
    digits
    space
    justify
    decimal
    leading_space : bool, optional, default True
        Whether the array should be formatted with a leading space.
        When an array as a column of a Series or DataFrame, we do want
        the leading space to pad between columns.

        When formatting an Index subclass
        (e.g. IntervalIndex._format_native_types), we don't want the
        leading space since it should be left-aligned.

    Returns
    -------
    List[str]
    """
    ...

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
    def get_result_as_array(self) -> np.ndarray:
        """
        Returns the float values converted into strings using
        the parameters given at initialisation, as a numpy array
        """
        ...

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
) -> list[str]:
    """
    Outputs rounded and formatted percentiles.

    Parameters
    ----------
    percentiles : list-like, containing floats from interval [0,1]

    Returns
    -------
    formatted : list of strings

    Notes
    -----
    Rounding precision is chosen so that: (1) if any two elements of
    ``percentiles`` differ, they remain different after rounding
    (2) no entry is *rounded* to 0% or 100%.
    Any non-integer is always rounded to at least 1 decimal place.

    Examples
    --------
    Keeps all entries different after rounding:

    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']

    No element is rounded to 0% or 100% (unless already equal to it).
    Duplicates are allowed:

    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
    """
    ...

def is_dates_only(values: np.ndarray | DatetimeArray | Index | DatetimeIndex) -> bool: ...
def get_format_datetime64(
    is_dates_only: bool, nat_rep: str = ..., date_format: str | None = ...
) -> Callable: ...
def get_format_datetime64_from_values(
    values: np.ndarray | DatetimeArray | DatetimeIndex, date_format: str | None
) -> str | None:
    """given values and a date_format, return a string format"""
    ...

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
) -> Callable:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
    ...

class EngFormatter:
    """
    Formats float values according to engineering format.

    Based on matplotlib.ticker.EngFormatter
    """

    ENG_PREFIXES = ...
    def __init__(
        self, accuracy: int | None = ..., use_eng_prefix: bool = ...
    ) -> None: ...
    def __call__(self, num: int | float) -> str:
        """
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:

        >>> format_eng(0)       # for self.accuracy = 0
        ' 0'

        >>> format_eng(1000000) # for self.accuracy = 1,
                                #     self.use_eng_prefix = True
        ' 1.0M'

        >>> format_eng("-1e-6") # for self.accuracy = 2
                                #     self.use_eng_prefix = False
        '-1.00E-06'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        """
        ...

def set_eng_float_format(accuracy: int = ..., use_eng_prefix: bool = ...) -> None:
    """
    Alter default behavior on how float is formatted in DataFrame.
    Format float in engineering format. By accuracy, we mean the number of
    decimal digits after the floating point.

    See also EngFormatter.
    """
    ...

def get_level_lengths(
    levels: Any, sentinel: bool | object | str = ...
) -> list[dict[int, int]]:
    """
    For each index in each level the function returns lengths of indexes.

    Parameters
    ----------
    levels : list of lists
        List of values on for level.
    sentinel : string, optional
        Value which states that no new index starts on there.

    Returns
    -------
    Returns list of maps. For each level returns map of indexes (key is index
    in row and value is length of index).
    """
    ...

def buffer_put_lines(buf: IO[str], lines: list[str]) -> None:
    """
    Appends lines to a buffer.

    Parameters
    ----------
    buf
        The buffer to write to
    lines
        The lines to append.
    """
    ...
