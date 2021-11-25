from typing import Any

from pandas._typing import StorageOptions
from pandas.io.excel._base import ExcelWriter

class _XlsxStyler:
    STYLE_MAPPING: dict[str, list[tuple[tuple[str, ...], str]]] = ...
    @classmethod
    def convert(cls, style_dict, num_format_str=...):
        """
        converts a style_dict to an xlsxwriter format dict

        Parameters
        ----------
        style_dict : style dictionary to convert
        num_format_str : optional number format string
        """
        ...

class XlsxWriter(ExcelWriter):
    engine = ...
    supported_extensions = ...
    def __init__(
        self,
        path,
        engine=...,
        date_format=...,
        datetime_format=...,
        mode: str = ...,
        storage_options: StorageOptions = ...,
        if_sheet_exists: str | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
        **kwargs
    ) -> None: ...
    def save(self):
        """
        Save workbook to disk.
        """
        ...
    def write_cells(
        self, cells, sheet_name=..., startrow=..., startcol=..., freeze_panes=...
    ): ...
