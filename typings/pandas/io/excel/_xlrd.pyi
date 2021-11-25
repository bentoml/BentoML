from pandas._typing import StorageOptions
from pandas.io.excel._base import BaseExcelReader

class XlrdReader(BaseExcelReader):
    def __init__(self, filepath_or_buffer, storage_options: StorageOptions = ...) -> None:
        """
        Reader using xlrd engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        storage_options : dict, optional
            passed to fsspec for appropriate URLs (see ``_get_filepath_or_buffer``)
        """
        ...
    def load_workbook(self, filepath_or_buffer): ...
    @property
    def sheet_names(self): ...
    def get_sheet_by_name(self, name): ...
    def get_sheet_by_index(self, index): ...
    def get_sheet_data(self, sheet, convert_float): ...
