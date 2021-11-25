def randbool(size=..., p: float = ...): ...

RANDS_CHARS = ...
RANDU_CHARS = ...

def rands_array(nchars, size, dtype=...):
    """
    Generate an array of byte strings.
    """
    ...

def randu_array(nchars, size, dtype=...):
    """
    Generate an array of unicode strings.
    """
    ...

def rands(nchars):  # -> str:
    """
    Generate one random byte string.

    See `rands_array` if you want to create an array of random strings.

    """
    ...
