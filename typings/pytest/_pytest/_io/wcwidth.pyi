
from functools import lru_cache

@lru_cache(100)
def wcwidth(c: str) -> int:
    """Determine how many columns are needed to display a character in a terminal.

    Returns -1 if the character is not printable.
    Returns 0, 1 or 2 for other characters.
    """
    ...

def wcswidth(s: str) -> int:
    """Determine how many columns are needed to display a string in a terminal.

    Returns -1 if the string contains non-printable characters.
    """
    ...

