
import pprint
import reprlib

class SafeRepr(reprlib.Repr):
    """repr.Repr that limits the resulting size of repr() and includes
    information on exceptions raised during the call."""
    def __init__(self, maxsize: int) -> None:
        ...
    
    def repr(self, x: object) -> str:
        ...
    
    def repr_instance(self, x: object, level: int) -> str:
        ...
    


def safeformat(obj: object) -> str:
    """Return a pretty printed string for the given object.

    Failing __repr__ functions of user instances will be represented
    with a short exception info.
    """
    ...

def saferepr(obj: object, maxsize: int = ...) -> str:
    """Return a size-limited safe repr-string for the given object.

    Failing __repr__ functions of user instances will be represented
    with a short exception info and 'saferepr' generally takes
    care to never raise exceptions itself.

    This function is a wrapper around the Repr/reprlib functionality of the
    standard 2.6 lib.
    """
    ...

class AlwaysDispatchingPrettyPrinter(pprint.PrettyPrinter):
    """PrettyPrinter that always dispatches (regardless of width)."""
    ...


