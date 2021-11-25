import datetime
from collections import abc
from typing import Any, Hashable, Sequence

import numpy as np
from pandas._typing import CompressionOptions, FilePathOrBuffer, StorageOptions
from pandas.core import generic
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.util._decorators import Appender, doc

"""
Module contains tools for processing Stata files into DataFrames

The StataReader below was originally written by Joe Presbrey as part of PyDTA.
It has been extended and improved by Skipper Seabold from the Statsmodels
project who also developed the StataWriter and was finally added to pandas in
a once again improved version.

You can find more information on http://presbrey.mit.edu/PyDTA and
https://www.statsmodels.org/devel/
"""
_version_error = ...
_statafile_processing_params1 = ...
_statafile_processing_params2 = ...
_chunksize_params = ...
_compression_params = ...
_iterator_params = ...
_reader_notes = ...
_read_stata_doc = ...
_read_method_doc = ...
_stata_reader_doc = ...
_date_formats = ...
stata_epoch = ...
excessive_string_length_error = ...

class PossiblePrecisionLoss(Warning): ...

precision_loss_doc = ...

class ValueLabelTypeMismatch(Warning): ...

value_label_mismatch_doc = ...

class InvalidColumnName(Warning): ...

invalid_name_doc = ...

class CategoricalConversionWarning(Warning): ...

categorical_conversion_warning = ...

class StataValueLabel:
    """
    Parse a categorical column and prepare formatted output

    Parameters
    ----------
    catarray : Series
        Categorical Series to encode
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(self, catarray: Series, encoding: str = ...) -> None: ...
    def generate_value_label(self, byteorder: str) -> bytes:
        """
        Generate the binary representation of the value labels.

        Parameters
        ----------
        byteorder : str
            Byte order of the output

        Returns
        -------
        value_label : bytes
            Bytes containing the formatted value label
        """
        ...

class StataMissingValue:
    """
    An observation's missing value.

    Parameters
    ----------
    value : {int, float}
        The Stata missing value code

    Notes
    -----
    More information: <https://www.stata.com/help.cgi?missing>

    Integer missing values make the code '.', '.a', ..., '.z' to the ranges
    101 ... 127 (for int8), 32741 ... 32767  (for int16) and 2147483621 ...
    2147483647 (for int32).  Missing values for floating point data types are
    more complex but the pattern is simple to discern from the following table.

    np.float32 missing values (float in Stata)
    0000007f    .
    0008007f    .a
    0010007f    .b
    ...
    00c0007f    .x
    00c8007f    .y
    00d0007f    .z

    np.float64 missing values (double in Stata)
    000000000000e07f    .
    000000000001e07f    .a
    000000000002e07f    .b
    ...
    000000000018e07f    .x
    000000000019e07f    .y
    00000000001ae07f    .z
    """

    MISSING_VALUES: dict[float, str] = ...
    bases = ...
    float32_base = ...
    increment = ...
    float64_base = ...
    increment = ...
    BASE_MISSING_VALUES = ...
    def __init__(self, value: int | float) -> None: ...
    @property
    def string(self) -> str:
        """
        The Stata representation of the missing value: '.', '.a'..'.z'

        Returns
        -------
        str
            The representation of the missing value.
        """
        ...
    @property
    def value(self) -> int | float:
        """
        The binary representation of the missing value.

        Returns
        -------
        {int, float}
            The binary representation of the missing value.
        """
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> int | float: ...

class StataParser:
    def __init__(self) -> None: ...

class StataReader(StataParser, abc.Iterator):
    __doc__ = ...
    def __init__(
        self,
        path_or_buf: FilePathOrBuffer,
        convert_dates: bool = ...,
        convert_categoricals: bool = ...,
        index_col: str | None = ...,
        convert_missing: bool = ...,
        preserve_dtypes: bool = ...,
        columns: Sequence[str] | None = ...,
        order_categoricals: bool = ...,
        chunksize: int | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def __enter__(self) -> StataReader:
        """enter context manager"""
        ...
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """exit context manager"""
        ...
    def close(self) -> None:
        """close the handle if its open"""
        ...
    def __next__(self) -> DataFrame: ...
    def get_chunk(self, size: int | None = ...) -> DataFrame:
        """
        Reads lines from Stata file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
        ...
    @Appender(_read_method_doc)
    def read(
        self,
        nrows: int | None = ...,
        convert_dates: bool | None = ...,
        convert_categoricals: bool | None = ...,
        index_col: str | None = ...,
        convert_missing: bool | None = ...,
        preserve_dtypes: bool | None = ...,
        columns: Sequence[str] | None = ...,
        order_categoricals: bool | None = ...,
    ) -> DataFrame: ...
    @property
    def data_label(self) -> str:
        """
        Return data label of Stata file.
        """
        ...
    def variable_labels(self) -> dict[str, str]:
        """
        Return variable labels as a dict, associating each variable name
        with corresponding label.

        Returns
        -------
        dict
        """
        ...
    def value_labels(self) -> dict[str, dict[float | int, str]]:
        """
        Return a dict, associating each variable name a dict, associating
        each value its corresponding label.

        Returns
        -------
        dict
        """
        ...

@Appender(_read_stata_doc)
def read_stata(
    filepath_or_buffer: FilePathOrBuffer,
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: Sequence[str] | None = ...,
    order_categoricals: bool = ...,
    chunksize: int | None = ...,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame | StataReader: ...
@doc(storage_options=generic._shared_docs["storage_options"])
class StataWriter(StataParser):
    """
    A class for writing Stata binary dta files

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    compression : str or dict, default 'infer'
        For on-the-fly compression of the output dta. If string, specifies
        compression mode. If dict, value at key 'method' specifies compression
        mode. Compression mode must be one of {{'infer', 'gzip', 'bz2', 'zip',
        'xz', None}}. If compression mode is 'infer' and `fname` is path-like,
        then detect compression from the following extensions: '.gz', '.bz2',
        '.zip', or '.xz' (otherwise no compression). If dict and compression
        mode is one of {{'zip', 'gzip', 'bz2'}}, or inferred as one of the above,
        other entries passed as additional compression options.

        .. versionadded:: 1.1.0

    {storage_options}

        .. versionadded:: 1.2.0

    Returns
    -------
    writer : StataWriter instance
        The StataWriter instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime.datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1]], columns=['a', 'b'])
    >>> writer = StataWriter('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {{"method": "zip", "archive_name": "data_file.dta"}}
    >>> writer = StataWriter('./data_file.zip', data, compression=compression)
    >>> writer.write_file()

    Save a DataFrame with dates
    >>> from datetime import datetime
    >>> data = pd.DataFrame([[datetime(2000,1,1)]], columns=['date'])
    >>> writer = StataWriter('./date_data_file.dta', data, {{'date' : 'tw'}})
    >>> writer.write_file()
    """

    _max_string_length = ...
    _encoding = ...
    def __init__(
        self,
        fname: FilePathOrBuffer,
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[Hashable, str] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def write_file(self) -> None: ...

class StataStrLWriter:
    """
    Converter for Stata StrLs

    Stata StrLs map 8 byte values to strings which are stored using a
    dictionary-like format where strings are keyed to two values.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert
    columns : Sequence[str]
        List of columns names to convert to StrL
    version : int, optional
        dta version.  Currently supports 117, 118 and 119
    byteorder : str, optional
        Can be ">", "<", "little", or "big". default is `sys.byteorder`

    Notes
    -----
    Supports creation of the StrL block of a dta file for dta versions
    117, 118 and 119.  These differ in how the GSO is stored.  118 and
    119 store the GSO lookup value as a uint32 and a uint64, while 117
    uses two uint32s. 118 and 119 also encode all strings as unicode
    which is required by the format.  117 uses 'latin-1' a fixed width
    encoding that extends the 7-bit ascii table with an additional 128
    characters.
    """

    def __init__(
        self,
        df: DataFrame,
        columns: Sequence[str],
        version: int = ...,
        byteorder: str | None = ...,
    ) -> None: ...
    def generate_table(self) -> tuple[dict[str, tuple[int, int]], DataFrame]:
        """
        Generates the GSO lookup table for the DataFrame

        Returns
        -------
        gso_table : dict
            Ordered dictionary using the string found as keys
            and their lookup position (v,o) as values
        gso_df : DataFrame
            DataFrame where strl columns have been converted to
            (v,o) values

        Notes
        -----
        Modifies the DataFrame in-place.

        The DataFrame returned encodes the (v,o) values as uint64s. The
        encoding depends on the dta version, and can be expressed as

        enc = v + o * 2 ** (o_size * 8)

        so that v is stored in the lower bits and o is in the upper
        bits. o_size is

          * 117: 4
          * 118: 6
          * 119: 5
        """
        ...
    def generate_blob(self, gso_table: dict[str, tuple[int, int]]) -> bytes:
        """
        Generates the binary blob of GSOs that is written to the dta file.

        Parameters
        ----------
        gso_table : dict
            Ordered dictionary (str, vo)

        Returns
        -------
        gso : bytes
            Binary content of dta file to be placed between strl tags

        Notes
        -----
        Output format depends on dta version.  117 uses two uint32s to
        express v and o while 118+ uses a uint32 for v and a uint64 for o.
        """
        ...

class StataWriter117(StataWriter):
    """
    A class for writing Stata binary dta files in Stata 13 format (117)

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    compression : str or dict, default 'infer'
        For on-the-fly compression of the output dta. If string, specifies
        compression mode. If dict, value at key 'method' specifies compression
        mode. Compression mode must be one of {'infer', 'gzip', 'bz2', 'zip',
        'xz', None}. If compression mode is 'infer' and `fname` is path-like,
        then detect compression from the following extensions: '.gz', '.bz2',
        '.zip', or '.xz' (otherwise no compression). If dict and compression
        mode is one of {'zip', 'gzip', 'bz2'}, or inferred as one of the above,
        other entries passed as additional compression options.

        .. versionadded:: 1.1.0

    Returns
    -------
    writer : StataWriter117 instance
        The StataWriter117 instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime.datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> from pandas.io.stata import StataWriter117
    >>> data = pd.DataFrame([[1.0, 1, 'a']], columns=['a', 'b', 'c'])
    >>> writer = StataWriter117('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = StataWriter117('./data_file.zip', data, compression=compression)
    >>> writer.write_file()

    Or with long strings stored in strl format
    >>> data = pd.DataFrame([['A relatively long string'], [''], ['']],
    ...                     columns=['strls'])
    >>> writer = StataWriter117('./data_file_with_long_strings.dta', data,
    ...                         convert_strl=['strls'])
    >>> writer.write_file()
    """

    _max_string_length = ...
    _dta_version = ...
    def __init__(
        self,
        fname: FilePathOrBuffer,
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[Hashable, str] | None = ...,
        convert_strl: Sequence[Hashable] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...

class StataWriterUTF8(StataWriter117):
    """
    Stata binary dta file writing in Stata 15 (118) and 16 (119) formats

    DTA 118 and 119 format files support unicode string data (both fixed
    and strL) format. Unicode is also supported in value labels, variable
    labels and the dataset label. Format 119 is automatically used if the
    file contains more than 32,767 variables.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict, default None
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool, default True
        Write the index to Stata dataset.
    byteorder : str, default None
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime, default None
        A datetime to use as file creation date.  Default is the current time
    data_label : str, default None
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict, default None
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list, default None
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    version : int, default None
        The dta version to use. By default, uses the size of data to determine
        the version. 118 is used if data.shape[1] <= 32767, and 119 is used
        for storing larger DataFrames.
    compression : str or dict, default 'infer'
        For on-the-fly compression of the output dta. If string, specifies
        compression mode. If dict, value at key 'method' specifies compression
        mode. Compression mode must be one of {'infer', 'gzip', 'bz2', 'zip',
        'xz', None}. If compression mode is 'infer' and `fname` is path-like,
        then detect compression from the following extensions: '.gz', '.bz2',
        '.zip', or '.xz' (otherwise no compression). If dict and compression
        mode is one of {'zip', 'gzip', 'bz2'}, or inferred as one of the above,
        other entries passed as additional compression options.

        .. versionadded:: 1.1.0

    Returns
    -------
    StataWriterUTF8
        The instance has a write_file method, which will write the file to the
        given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime.datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    Using Unicode data and column names

    >>> from pandas.io.stata import StataWriterUTF8
    >>> data = pd.DataFrame([[1.0, 1, 'ᴬ']], columns=['a', 'β', 'ĉ'])
    >>> writer = StataWriterUTF8('./data_file.dta', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = StataWriterUTF8('./data_file.zip', data, compression=compression)
    >>> writer.write_file()

    Or with long strings stored in strl format

    >>> data = pd.DataFrame([['ᴀ relatively long ŝtring'], [''], ['']],
    ...                     columns=['strls'])
    >>> writer = StataWriterUTF8('./data_file_with_long_strings.dta', data,
    ...                          convert_strl=['strls'])
    >>> writer.write_file()
    """

    _encoding = ...
    def __init__(
        self,
        fname: FilePathOrBuffer,
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[Hashable, str] | None = ...,
        convert_strl: Sequence[Hashable] | None = ...,
        version: int | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
