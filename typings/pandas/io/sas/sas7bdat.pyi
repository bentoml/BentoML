from collections import abc

import numpy as np
from pandas import DataFrame
from pandas.io.sas.sasreader import ReaderBase

"""
Read SAS7BDAT files

Based on code written by Jared Hobbs:
  https://bitbucket.org/jaredhobbs/sas7bdat

See also:
  https://github.com/BioStatMatt/sas7bdat

Partial documentation of the file format:
  https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf

Reference for binary data compression:
  http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
"""

class _SubheaderPointer:
    offset: int
    length: int
    compression: int
    ptype: int
    def __init__(
        self, offset: int, length: int, compression: int, ptype: int
    ) -> None: ...

class _Column:
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes
    ctype: bytes
    length: int
    def __init__(
        self,
        col_id: int,
        name: str | bytes,
        label: str | bytes,
        format: str | bytes,
        ctype: bytes,
        length: int,
    ) -> None: ...

class SAS7BDATReader(ReaderBase, abc.Iterator):
    """
    Read SAS files in SAS7BDAT format.

    Parameters
    ----------
    path_or_buf : path name or buffer
        Name of SAS file or file-like object pointing to SAS file
        contents.
    index : column identifier, defaults to None
        Column to use as index.
    convert_dates : bool, defaults to True
        Attempt to convert dates to Pandas datetime values.  Note that
        some rarely used SAS date formats may be unsupported.
    blank_missing : bool, defaults to True
        Convert empty strings to missing values (SAS uses blanks to
        indicate missing character variables).
    chunksize : int, defaults to None
        Return SAS7BDATReader object for iterations, returns chunks
        with given number of lines.
    encoding : string, defaults to None
        String encoding.
    convert_text : bool, defaults to True
        If False, text variables are left as raw bytes.
    convert_header_text : bool, defaults to True
        If False, header text, including column names, are left as raw
        bytes.
    """

    _int_length: int
    _cached_page: bytes | None
    def __init__(
        self,
        path_or_buf,
        index=...,
        convert_dates=...,
        blank_missing=...,
        chunksize=...,
        encoding=...,
        convert_text=...,
        convert_header_text=...,
    ) -> None: ...
    def column_data_lengths(self) -> np.ndarray:
        """Return a numpy int64 array of the column data lengths"""
        ...
    def column_data_offsets(self) -> np.ndarray:
        """Return a numpy int64 array of the column offsets"""
        ...
    def column_types(self) -> np.ndarray:
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
        ...
    def close(self) -> None: ...
    def __next__(self): ...
    def read(self, nrows: int | None = ...) -> DataFrame | None: ...
