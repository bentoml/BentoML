"""This module is designed for community supported date conversion functions"""

def parse_date_time(date_col, time_col):  # -> ndarray:
    """
    Parse columns with dates and times into a single datetime column.

    .. deprecated:: 1.2
    """
    ...

def parse_date_fields(year_col, month_col, day_col):  # -> ndarray:
    """
    Parse columns with years, months and days into a single date column.

    .. deprecated:: 1.2
    """
    ...

def parse_all_fields(
    year_col, month_col, day_col, hour_col, minute_col, second_col
):  # -> ndarray:
    """
    Parse columns with datetime information into a single datetime column.

    .. deprecated:: 1.2
    """
    ...

def generic_parser(parse_func, *cols):  # -> ndarray:
    """
    Use dateparser to parse columns with data information into a single datetime column.

    .. deprecated:: 1.2
    """
    ...
