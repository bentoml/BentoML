from typing import Any

from pandas._typing import StorageOptions
from pandas.io.excel._base import ExcelWriter
from pandas.io.formats.excel import ExcelCell

class ODSWriter(ExcelWriter):
    engine = ...
    supported_extensions = ...
    def __init__(
        self,
        path: str,
        engine: str | None = ...,
        date_format=...,
        datetime_format=...,
        mode: str = ...,
        storage_options: StorageOptions = ...,
        if_sheet_exists: str | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
        **kwargs
    ) -> None: ...
    def save(self) -> None:
        """
        Save workbook to disk.
        """
        ...
    def write_cells(
        self,
        cells: list[ExcelCell],
        sheet_name: str | None = ...,
        startrow: int = ...,
        startcol: int = ...,
        freeze_panes: tuple[int, int] | None = ...,
    ) -> None:
        """
        Write the frame cells using odf
        """
        ...
