from typing import Callable, Hashable, Iterable, Mapping, Sequence

from pandas._typing import IndexLabel, StorageOptions
from pandas.core import generic
from pandas.util._decorators import doc

"""
Utilities for conversion to writer-agnostic Excel representation.
"""

class ExcelCell:
    __fields__ = ...
    __slots__ = ...
    def __init__(
        self,
        row: int,
        col: int,
        val,
        style=...,
        mergestart: int | None = ...,
        mergeend: int | None = ...,
    ) -> None: ...

class CSSToExcelConverter:
    """
    A callable for converting CSS declarations to ExcelWriter styles

    Supports parts of CSS 2.2, with minimal CSS 3.0 support (e.g. text-shadow),
    focusing on font styling, backgrounds, borders and alignment.

    Operates by first computing CSS styles in a fairly generic
    way (see :meth:`compute_css`) then determining Excel style
    properties from CSS properties (see :meth:`build_xlstyle`).

    Parameters
    ----------
    inherited : str, optional
        CSS declarations understood to be the containing scope for the
        CSS processed by :meth:`__call__`.
    """

    NAMED_COLORS = ...
    VERTICAL_MAP = ...
    BOLD_MAP = ...
    ITALIC_MAP = ...
    FAMILY_MAP = ...
    inherited: dict[str, str] | None
    def __init__(self, inherited: str | None = ...) -> None: ...
    compute_css = ...
    def __call__(self, declarations_str: str) -> dict[str, dict[str, str]]:
        """
        Convert CSS declarations to ExcelWriter style.

        Parameters
        ----------
        declarations_str : str
            List of CSS declarations.
            e.g. "font-weight: bold; background: blue"

        Returns
        -------
        xlstyle : dict
            A style as interpreted by ExcelWriter when found in
            ExcelCell.style.
        """
        ...
    def build_xlstyle(self, props: Mapping[str, str]) -> dict[str, dict[str, str]]: ...
    def build_alignment(
        self, props: Mapping[str, str]
    ) -> dict[str, bool | str | None]: ...
    def build_border(
        self, props: Mapping[str, str]
    ) -> dict[str, dict[str, str | None]]: ...
    def build_fill(self, props: Mapping[str, str]): ...
    def build_number_format(self, props: Mapping[str, str]) -> dict[str, str | None]: ...
    def build_font(
        self, props: Mapping[str, str]
    ) -> dict[str, bool | int | float | str | None]: ...
    def color_to_excel(self, val: str | None) -> str | None: ...

class ExcelFormatter:
    """
    Class for formatting a DataFrame to a list of ExcelCells,

    Parameters
    ----------
    df : DataFrame or Styler
    na_rep: na representation
    float_format : str, default None
        Format string for floating point numbers
    cols : sequence, optional
        Columns to write
    header : bool or sequence of str, default True
        Write out column names. If a list of string is given it is
        assumed to be aliases for the column names
    index : bool, default True
        output row names (index)
    index_label : str or sequence, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the DataFrame uses MultiIndex.
    merge_cells : bool, default False
        Format MultiIndex and Hierarchical Rows as merged cells.
    inf_rep : str, default `'inf'`
        representation for np.inf values (which aren't representable in Excel)
        A `'-'` sign will be added in front of -inf.
    style_converter : callable, optional
        This translates Styler styles (CSS) into ExcelWriter styles.
        Defaults to ``CSSToExcelConverter()``.
        It should have signature css_declarations string -> excel style.
        This is only called for body cells.
    """

    max_rows = 2 ** 20
    max_cols = 2 ** 14
    def __init__(
        self,
        df,
        na_rep: str = ...,
        float_format: str | None = ...,
        cols: Sequence[Hashable] | None = ...,
        header: Sequence[Hashable] | bool = ...,
        index: bool = ...,
        index_label: IndexLabel | None = ...,
        merge_cells: bool = ...,
        inf_rep: str = ...,
        style_converter: Callable | None = ...,
    ) -> None: ...
    @property
    def header_style(self): ...
    def get_formatted_cells(self) -> Iterable[ExcelCell]: ...
    @doc(storage_options=generic._shared_docs["storage_options"])
    def write(
        self,
        writer,
        sheet_name=...,
        startrow=...,
        startcol=...,
        freeze_panes=...,
        engine=...,
        storage_options: StorageOptions = ...,
    ):  # -> None:
        """
        writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame
        startrow :
            upper left cell row to dump data frame
        startcol :
            upper left cell column to dump data frame
        freeze_panes : tuple of integer (length 2), default None
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen
        engine : string, default None
            write engine to use if writer is a path - you can also set this
            via the options ``io.excel.xlsx.writer``, ``io.excel.xls.writer``,
            and ``io.excel.xlsm.writer``.

            .. deprecated:: 1.2.0

                As the `xlwt <https://pypi.org/project/xlwt/>`__ package is no longer
                maintained, the ``xlwt`` engine will be removed in a future
                version of pandas.

        {storage_options}

            .. versionadded:: 1.2.0
        """
        ...
