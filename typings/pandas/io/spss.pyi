from pathlib import Path
from typing import Sequence
from pandas.core.api import DataFrame

def read_spss(
    path: str | Path,
    usecols: Sequence[str] | None = ...,
    convert_categoricals: bool = ...,
) -> DataFrame: ...
