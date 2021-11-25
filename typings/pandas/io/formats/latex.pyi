from abc import ABC, abstractmethod
from typing import Iterator

from pandas.io.formats.format import DataFrameFormatter

"""
Module for formatting output data in Latex.
"""

class RowStringConverter(ABC):
    r"""Converter for dataframe rows into LaTeX strings.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
        Instance of `DataFrameFormatter`.
    multicolumn: bool, optional
        Whether to use \multicolumn macro.
    multicolumn_format: str, optional
        Multicolumn format.
    multirow: bool, optional
        Whether to use \multirow macro.

    """
    def __init__(
        self,
        formatter: DataFrameFormatter,
        multicolumn: bool = ...,
        multicolumn_format: str | None = ...,
        multirow: bool = ...,
    ) -> None: ...
    def get_strrow(self, row_num: int) -> str:
        """Get string representation of the row."""
        ...
    @property
    def index_levels(self) -> int:
        """Integer number of levels in index."""
        ...
    @property
    def column_levels(self) -> int: ...
    @property
    def header_levels(self) -> int: ...

class RowStringIterator(RowStringConverter):
    """Iterator over rows of the header or the body of the table."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over LaTeX string representations of rows."""
        ...

class RowHeaderIterator(RowStringIterator):
    """Iterator for the table header rows."""

    def __iter__(self) -> Iterator[str]: ...

class RowBodyIterator(RowStringIterator):
    """Iterator for the table body rows."""

    def __iter__(self) -> Iterator[str]: ...

class TableBuilderAbstract(ABC):
    """
    Abstract table builder producing string representation of LaTeX table.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
        Instance of `DataFrameFormatter`.
    column_format: str, optional
        Column format, for example, 'rcl' for three columns.
    multicolumn: bool, optional
        Use multicolumn to enhance MultiIndex columns.
    multicolumn_format: str, optional
        The alignment for multicolumns, similar to column_format.
    multirow: bool, optional
        Use multirow to enhance MultiIndex rows.
    caption: str, optional
        Table caption.
    short_caption: str, optional
        Table short caption.
    label: str, optional
        LaTeX label.
    position: str, optional
        Float placement specifier, for example, 'htb'.
    """

    def __init__(
        self,
        formatter: DataFrameFormatter,
        column_format: str | None = ...,
        multicolumn: bool = ...,
        multicolumn_format: str | None = ...,
        multirow: bool = ...,
        caption: str | None = ...,
        short_caption: str | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> None: ...
    def get_result(self) -> str:
        """String representation of LaTeX table."""
        ...
    @property
    @abstractmethod
    def env_begin(self) -> str:
        """Beginning of the environment."""
        ...
    @property
    @abstractmethod
    def top_separator(self) -> str:
        """Top level separator."""
        ...
    @property
    @abstractmethod
    def header(self) -> str:
        """Header lines."""
        ...
    @property
    @abstractmethod
    def middle_separator(self) -> str:
        """Middle level separator."""
        ...
    @property
    @abstractmethod
    def env_body(self) -> str:
        """Environment body."""
        ...
    @property
    @abstractmethod
    def bottom_separator(self) -> str:
        """Bottom level separator."""
        ...
    @property
    @abstractmethod
    def env_end(self) -> str:
        """End of the environment."""
        ...

class GenericTableBuilder(TableBuilderAbstract):
    """Table builder producing string representation of LaTeX table."""

    @property
    def header(self) -> str: ...
    @property
    def top_separator(self) -> str: ...
    @property
    def middle_separator(self) -> str: ...
    @property
    def env_body(self) -> str: ...

class LongTableBuilder(GenericTableBuilder):
    """Concrete table builder for longtable.

    >>> from pandas import DataFrame
    >>> from pandas.io.formats import format as fmt
    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = LongTableBuilder(formatter, caption='a long table',
    ...                            label='tab:long', column_format='lrl')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{longtable}{lrl}
    \\caption{a long table}
    \\label{tab:long}\\\\
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    \\endfirsthead
    \\caption[]{a long table} \\\\
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    \\endhead
    \\midrule
    \\multicolumn{3}{r}{{Continued on next page}} \\\\
    \\midrule
    \\endfoot
    <BLANKLINE>
    \\bottomrule
    \\endlastfoot
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\end{longtable}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str: ...
    @property
    def middle_separator(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class RegularTableBuilder(GenericTableBuilder):
    """Concrete table builder for regular table.

    >>> from pandas import DataFrame
    >>> from pandas.io.formats import format as fmt
    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = RegularTableBuilder(formatter, caption='caption', label='lab',
    ...                               column_format='lrc')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{table}
    \\centering
    \\caption{caption}
    \\label{lab}
    \\begin{tabular}{lrc}
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\bottomrule
    \\end{tabular}
    \\end{table}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class TabularBuilder(GenericTableBuilder):
    """Concrete table builder for tabular environment.

    >>> from pandas import DataFrame
    >>> from pandas.io.formats import format as fmt
    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = TabularBuilder(formatter, column_format='lrc')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{tabular}{lrc}
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\bottomrule
    \\end{tabular}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class LatexFormatter:
    r"""
    Used to render a DataFrame to a LaTeX tabular/longtable environment output.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
    longtable : bool, default False
        Use longtable environment.
    column_format : str, default None
        The columns format as specified in `LaTeX table format
        <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl' for 3 columns
    multicolumn : bool, default False
        Use \multicolumn to enhance MultiIndex columns.
    multicolumn_format : str, default 'l'
        The alignment for multicolumns, similar to `column_format`
    multirow : bool, default False
        Use \multirow to enhance MultiIndex rows.
    caption : str or tuple, optional
        Tuple (full_caption, short_caption),
        which results in \caption[short_caption]{full_caption};
        if a single string is passed, no short caption will be set.
    label : str, optional
        The LaTeX label to be placed inside ``\label{}`` in the output.
    position : str, optional
        The LaTeX positional argument for tables, to be placed after
        ``\begin{}`` in the output.

    See Also
    --------
    HTMLFormatter
    """
    def __init__(
        self,
        formatter: DataFrameFormatter,
        longtable: bool = ...,
        column_format: str | None = ...,
        multicolumn: bool = ...,
        multicolumn_format: str | None = ...,
        multirow: bool = ...,
        caption: str | tuple[str, str] | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> None: ...
    def to_string(self) -> str:
        """
        Render a DataFrame to a LaTeX tabular, longtable, or table/tabular
        environment output.
        """
        ...
    @property
    def builder(self) -> TableBuilderAbstract:
        """Concrete table builder.

        Returns
        -------
        TableBuilder
        """
        ...
    @property
    def column_format(self) -> str | None:
        """Column format."""
        ...
    @column_format.setter
    def column_format(self, input_column_format: str | None) -> None:
        """Setter for column format."""
        ...

if __name__ == "__main__": ...
