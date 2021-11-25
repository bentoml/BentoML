from typing import TYPE_CHECKING, Any

from pandas._typing import StorageOptions
from pandas.io.excel._base import ExcelWriter

if TYPE_CHECKING: ...

class XlwtWriter(ExcelWriter):
    engine = ...
    supported_extensions = ...
    def __init__(
        self,
        path,
        engine=...,
        date_format=...,
        datetime_format=...,
        encoding=...,
        mode: str = ...,
        storage_options: StorageOptions = ...,
        if_sheet_exists: str | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
        **kwargs
    ) -> None: ...
    def save(self):  # -> None:
        """
        Save workbook to disk.
        """
        ...
    def write_cells(
        self, cells, sheet_name=..., startrow=..., startcol=..., freeze_panes=...
    ): ...
