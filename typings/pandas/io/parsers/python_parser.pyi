from collections import abc
from typing import Iterator

from pandas._typing import FilePathOrBuffer
from pandas.io.parsers.base_parser import ParserBase

_BOM = ...

class PythonParser(ParserBase):
    def __init__(self, f: FilePathOrBuffer | list, **kwds) -> None:
        """
        Workhorse function for processing nested list into DataFrame
        """
        ...
    def read(self, rows=...): ...
    def get_chunk(self, size=...): ...
    _implicit_index = ...

class FixedWidthReader(abc.Iterator):
    """
    A reader of fixed-width lines.
    """

    def __init__(
        self, f, colspecs, delimiter, comment, skiprows=..., infer_nrows=...
    ) -> None: ...
    def get_rows(self, infer_nrows, skiprows=...):  # -> list[Unknown]:
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        ...
    def detect_colspecs(self, infer_nrows=..., skiprows=...): ...
    def __next__(self): ...

class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f, **kwds) -> None: ...

def count_empty_vals(vals) -> int: ...
