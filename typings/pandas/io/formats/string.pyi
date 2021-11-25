from pandas.io.formats.format import DataFrameFormatter

"""
Module for formatting output data in console (to string).
"""

class StringFormatter:
    """Formatter for string representation of a dataframe."""

    def __init__(self, fmt: DataFrameFormatter, line_width: int | None = ...) -> None: ...
    def to_string(self) -> str: ...
