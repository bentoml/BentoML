"""
Helpers for logging.

This module needs much love to become useful.
"""

def format_time(t): ...
def short_format_time(t): ...
def pformat(obj, indent=..., depth=...): ...

class Logger:
    """Base class for logging messages."""

    def __init__(self, depth=...) -> None:
        """
        Parameters
        ----------
        depth: int, optional
            The depth of objects printed.
        """
        ...
    def warn(self, msg): ...
    def debug(self, msg): ...
    def format(self, obj, indent=...):  # -> str:
        """Return the formatted representation of the object."""
        ...

class PrintTime:
    """Print and log messages while keeping track of time."""

    def __init__(self, logfile=..., logdir=...) -> None: ...
    def __call__(self, msg=..., total=...):  # -> None:
        """Print the time elapsed between the last call and the current
        call, with an optional message.
        """
        ...
