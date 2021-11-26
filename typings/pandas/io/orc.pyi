from typing import TYPE_CHECKING
from pandas import DataFrame
from pandas._typing import FilePathOrBuffer

if TYPE_CHECKING: ...

def read_orc(
    path: FilePathOrBuffer, columns: list[str] | None = ..., **kwargs
) -> DataFrame: ...
