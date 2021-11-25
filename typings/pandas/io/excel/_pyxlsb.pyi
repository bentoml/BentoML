from pandas._typing import FilePathOrBuffer, Scalar, StorageOptions
from pandas.io.excel._base import BaseExcelReader

class PyxlsbReader(BaseExcelReader):
    def __init__(
        self, filepath_or_buffer: FilePathOrBuffer, storage_options: StorageOptions = ...
    ) -> None:
        """
        Reader using pyxlsb engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object, or Workbook
            Object to be parsed.
        storage_options : dict, optional
            passed to fsspec for appropriate URLs (see ``_get_filepath_or_buffer``)
        """
        ...
    def load_workbook(self, filepath_or_buffer: FilePathOrBuffer): ...
    @property
    def sheet_names(self) -> list[str]: ...
    def get_sheet_by_name(self, name: str): ...
    def get_sheet_by_index(self, index: int): ...
    def get_sheet_data(self, sheet, convert_float: bool) -> list[list[Scalar]]: ...
