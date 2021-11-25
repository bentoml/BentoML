from pathlib import Path
from typing import Sequence

from pandas.core.api import DataFrame

def read_spss(
    path: str | Path,
    usecols: Sequence[str] | None = ...,
    convert_categoricals: bool = ...,
) -> DataFrame:
    """
    Load an SPSS file from the file path, returning a DataFrame.

    .. versionadded:: 0.25.0

    Parameters
    ----------
    path : str or Path
        File path.
    usecols : list-like, optional
        Return a subset of the columns. If None, return all columns.
    convert_categoricals : bool, default is True
        Convert categorical columns into pd.Categorical.

    Returns
    -------
    DataFrame
    """
    ...
