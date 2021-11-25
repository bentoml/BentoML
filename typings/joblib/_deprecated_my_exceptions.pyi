"""
Exceptions

This module is deprecated and will be removed in joblib 0.16.
"""

class JoblibException(Exception):
    """A simple exception with an error message that you can get to."""

    def __init__(self, *args) -> None: ...
    def __repr__(self): ...
    __str__ = ...

class TransportableException(JoblibException):
    """An exception containing all the info to wrap an original
    exception and recreate it.
    """

    def __init__(self, message, etype) -> None: ...
    def unwrap(self, context_message=...): ...

_exception_mapping = ...
