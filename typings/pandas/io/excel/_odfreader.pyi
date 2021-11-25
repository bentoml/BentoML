from pandas._typing import FilePathOrBuffer, Scalar, StorageOptions
from pandas.io.excel._base import BaseExcelReader

class ODFReader(BaseExcelReader):
    """
    Read tables out of OpenDocument formatted files.

    Parameters
    ----------
    filepath_or_buffer : str, path to be parsed or
        an open readable stream.
    storage_options : dict, optional
        passed to fsspec for appropriate URLs (see ``_get_filepath_or_buffer``)
    """

    def __init__(
        self, filepath_or_buffer: FilePathOrBuffer, storage_options: StorageOptions = ...
    ) -> None: ...
    def load_workbook(self, filepath_or_buffer: FilePathOrBuffer): ...
    @property
    def empty_value(self) -> str:
        """Property for compat with other readers."""
        ...
    @property
    def sheet_names(self) -> list[str]:
        """Return a list of sheet names present in the document"""
        ...
    def get_sheet_by_index(self, index: int): ...
    def get_sheet_by_name(self, name: str): ...
    def get_sheet_data(self, sheet, convert_float: bool) -> list[list[Scalar]]:
        """
        Parse an ODF Table into a list of lists
        """
        ...
