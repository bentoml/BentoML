import pandas._libs.parsers as parsers
from pandas._typing import FilePathOrBuffer
from pandas.io.parsers.base_parser import ParserBase

class CParserWrapper(ParserBase):
    low_memory: bool
    _reader: parsers.TextReader
    def __init__(self, src: FilePathOrBuffer, **kwds) -> None: ...
    def close(self) -> None: ...
    def read(self, nrows=...): ...

def ensure_dtype_objs(dtype):  # -> DtypeObj | dict[Unknown, DtypeObj]:
    """
    Ensure we have either None, a dtype object, or a dictionary mapping to
    dtype objects.
    """
    ...
