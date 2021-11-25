from typing import Any, Iterable

from pandas.io.formats.format import DataFrameFormatter

"""
Module for formatting output data in HTML.
"""

class HTMLFormatter:
    """
    Internal class for formatting output data in html.
    This class is intended for shared functionality between
    DataFrame.to_html() and DataFrame._repr_html_().
    Any logic in common with other output formatting methods
    should ideally be inherited from classes in format.py
    and this class responsible for only producing html markup.
    """

    indent_delta = ...
    def __init__(
        self,
        formatter: DataFrameFormatter,
        classes: str | list[str] | tuple[str, ...] | None = ...,
        border: int | None = ...,
        table_id: str | None = ...,
        render_links: bool = ...,
    ) -> None: ...
    def to_string(self) -> str: ...
    def render(self) -> list[str]: ...
    @property
    def should_show_dimensions(self): ...
    @property
    def show_row_idx_names(self) -> bool: ...
    @property
    def show_col_idx_names(self) -> bool: ...
    @property
    def row_levels(self) -> int: ...
    @property
    def is_truncated(self) -> bool: ...
    @property
    def ncols(self) -> int: ...
    def write(self, s: Any, indent: int = ...) -> None: ...
    def write_th(
        self, s: Any, header: bool = ..., indent: int = ..., tags: str | None = ...
    ) -> None:
        """
        Method for writing a formatted <th> cell.

        If col_space is set on the formatter then that is used for
        the value of min-width.

        Parameters
        ----------
        s : object
            The data to be written inside the cell.
        header : bool, default False
            Set to True if the <th> is for use inside <thead>.  This will
            cause min-width to be set if there is one.
        indent : int, default 0
            The indentation level of the cell.
        tags : str, default None
            Tags to include in the cell.

        Returns
        -------
        A written <th> cell.
        """
        ...
    def write_td(self, s: Any, indent: int = ..., tags: str | None = ...) -> None: ...
    def write_tr(
        self,
        line: Iterable,
        indent: int = ...,
        indent_delta: int = ...,
        header: bool = ...,
        align: str | None = ...,
        tags: dict[int, str] | None = ...,
        nindex_levels: int = ...,
    ) -> None: ...

class NotebookFormatter(HTMLFormatter):
    """
    Internal class for formatting output data in html for display in Jupyter
    Notebooks. This class is intended for functionality specific to
    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)
    """

    def write_style(self) -> None: ...
    def render(self) -> list[str]: ...
