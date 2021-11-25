from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Iterable, Mapping, Sequence

from pandas._typing import Dtype, FrameOrSeriesUnion
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index

if TYPE_CHECKING: ...

class BaseInfo(ABC):
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    """

    data: FrameOrSeriesUnion
    memory_usage: bool | str
    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """
        ...
    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
        ...
    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
        ...
    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """
        ...
    @property
    def memory_usage_string(self) -> str:
        """Memory usage in a form of human readable string."""
        ...
    @property
    def size_qualifier(self) -> str: ...
    @abstractmethod
    def render(
        self,
        *,
        buf: IO[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None
    ) -> None:
        """
        Print a concise summary of a %(klass)s.

        This method prints information about a %(klass)s including
        the index dtype%(type_sub)s, non-null values and memory usage.
        %(version_added_sub)s\

        Parameters
        ----------
        data : %(klass)s
            %(klass)s to print information about.
        verbose : bool, optional
            Whether to print the full summary. By default, the setting in
            ``pandas.options.display.max_info_columns`` is followed.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        %(max_cols_sub)s
        memory_usage : bool, str, optional
            Specifies whether total memory usage of the %(klass)s
            elements (including the index) should be displayed. By default,
            this follows the ``pandas.options.display.memory_usage`` setting.

            True always show memory usage. False never shows memory usage.
            A value of 'deep' is equivalent to "True with deep introspection".
            Memory usage is shown in human-readable units (base-2
            representation). Without deep introspection a memory estimation is
            made based in column dtype and number of rows assuming values
            consume the same memory amount for corresponding dtypes. With deep
            memory introspection, a real memory usage calculation is performed
            at the cost of computational resources.
        %(show_counts_sub)s

        Returns
        -------
        None
            This method prints a summary of a %(klass)s and returns None.

        See Also
        --------
        %(see_also_sub)s

        Examples
        --------
        %(examples_sub)s
        """
        ...

class DataFrameInfo(BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(
        self, data: DataFrame, memory_usage: bool | str | None = ...
    ) -> None: ...
    @property
    def dtype_counts(self) -> Mapping[str, int]: ...
    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        ...
    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        ...
    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        ...
    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
        ...
    @property
    def memory_usage_bytes(self) -> int: ...
    def render(
        self,
        *,
        buf: IO[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None
    ) -> None: ...

class InfoPrinterAbstract:
    """
    Class for printing dataframe or series info.
    """

    def to_buffer(self, buf: IO[str] | None = ...) -> None:
        """Save dataframe info into buffer."""
        ...

class DataFrameInfoPrinter(InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(
        self,
        info: DataFrameInfo,
        max_cols: int | None = ...,
        verbose: bool | None = ...,
        show_counts: bool | None = ...,
    ) -> None: ...
    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
        ...
    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
        ...
    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
        ...
    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        ...

class TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """

    _lines: list[str]
    info: BaseInfo
    @abstractmethod
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""
        ...
    @property
    def data(self) -> FrameOrSeriesUnion: ...
    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
        ...
    @property
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
        ...
    @property
    def display_memory_usage(self) -> bool:
        """Whether to display memory usage."""
        ...
    @property
    def memory_usage_string(self) -> str:
        """Memory usage string with proper size qualifier."""
        ...
    @property
    def non_null_counts(self) -> Sequence[int]: ...
    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
        ...
    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
        ...
    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
        ...

class DataFrameTableBuilder(TableBuilderAbstract):
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.
    """

    def __init__(self, *, info: DataFrameInfo) -> None: ...
    def get_lines(self) -> list[str]: ...
    @property
    def data(self) -> DataFrame:
        """DataFrame."""
        ...
    @property
    def ids(self) -> Index:
        """Dataframe columns."""
        ...
    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
        ...
    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        ...

class DataFrameTableBuilderNonVerbose(DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """

    def add_columns_summary_line(self) -> None: ...

class TableBuilderVerboseMixin(TableBuilderAbstract):
    """
    Mixin for verbose info output.
    """

    SPACING: str = ...
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool
    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        ...
    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
        ...
    def add_header_line(self) -> None: ...
    def add_separator_line(self) -> None: ...
    def add_body_lines(self) -> None: ...

class DataFrameTableBuilderVerbose(DataFrameTableBuilder, TableBuilderVerboseMixin):
    """
    Dataframe info table builder for verbose output.
    """

    def __init__(self, *, info: DataFrameInfo, with_counts: bool) -> None: ...
    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        ...
    def add_columns_summary_line(self) -> None: ...
