from abc import ABC, abstractmethod
from typing import Iterator
from pandas.io.formats.format import DataFrameFormatter

class RowStringConverter(ABC):
    def __init__(
        self,
        formatter: DataFrameFormatter,
        multicolumn: bool = ...,
        multicolumn_format: str | None = ...,
        multirow: bool = ...,
    ) -> None: ...
    def get_strrow(self, row_num: int) -> str: ...
    @property
    def index_levels(self) -> int: ...
    @property
    def column_levels(self) -> int: ...
    @property
    def header_levels(self) -> int: ...

class RowStringIterator(RowStringConverter):
    @abstractmethod
    def __iter__(self) -> Iterator[str]: ...

class RowHeaderIterator(RowStringIterator):
    def __iter__(self) -> Iterator[str]: ...

class RowBodyIterator(RowStringIterator):
    def __iter__(self) -> Iterator[str]: ...

class TableBuilderAbstract(ABC):
    def __init__(
        self,
        formatter: DataFrameFormatter,
        column_format: str | None = ...,
        multicolumn: bool = ...,
        multicolumn_format: str | None = ...,
        multirow: bool = ...,
        caption: str | None = ...,
        short_caption: str | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> None: ...
    def get_result(self) -> str: ...
    @property
    @abstractmethod
    def env_begin(self) -> str: ...
    @property
    @abstractmethod
    def top_separator(self) -> str: ...
    @property
    @abstractmethod
    def header(self) -> str: ...
    @property
    @abstractmethod
    def middle_separator(self) -> str: ...
    @property
    @abstractmethod
    def env_body(self) -> str: ...
    @property
    @abstractmethod
    def bottom_separator(self) -> str: ...
    @property
    @abstractmethod
    def env_end(self) -> str: ...

class GenericTableBuilder(TableBuilderAbstract):
    @property
    def header(self) -> str: ...
    @property
    def top_separator(self) -> str: ...
    @property
    def middle_separator(self) -> str: ...
    @property
    def env_body(self) -> str: ...

class LongTableBuilder(GenericTableBuilder):
    @property
    def env_begin(self) -> str: ...
    @property
    def middle_separator(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class RegularTableBuilder(GenericTableBuilder):
    @property
    def env_begin(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class TabularBuilder(GenericTableBuilder):
    @property
    def env_begin(self) -> str: ...
    @property
    def bottom_separator(self) -> str: ...
    @property
    def env_end(self) -> str: ...

class LatexFormatter:
    def __init__(
        self,
        formatter: DataFrameFormatter,
        longtable: bool = ...,
        column_format: str | None = ...,
        multicolumn: bool = ...,
        multicolumn_format: str | None = ...,
        multirow: bool = ...,
        caption: str | tuple[str, str] | None = ...,
        label: str | None = ...,
        position: str | None = ...,
    ) -> None: ...
    def to_string(self) -> str: ...
    @property
    def builder(self) -> TableBuilderAbstract: ...
    @property
    def column_format(self) -> str | None: ...
    @column_format.setter
    def column_format(self, input_column_format: str | None) -> None: ...

if __name__ == "__main__": ...
