"""
Internal module for console introspection
"""

def get_console_size():  # -> tuple[Unknown | int | None, Unknown | int | None]:
    """
    Return console size as tuple = (width, height).

    Returns (None,None) in non-interactive session.
    """
    ...

def in_interactive_session():  # -> Literal[True]:
    """
    Check if we're running in an interactive shell.

    Returns
    -------
    bool
        True if running under python/ipython interactive shell.
    """
    ...

def in_ipython_frontend():  # -> bool:
    """
    Check if we're inside an IPython zmq frontend.

    Returns
    -------
    bool
    """
    ...
