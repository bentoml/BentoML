
from typing import Any, Callable, List, Optional

"""Utilities for assertion debugging."""
_reprcompare: Optional[Callable[[str, object, object], Optional[str]]] = ...
_assertion_pass: Optional[Callable[[int, str, str], None]] = ...
def format_explanation(explanation: str) -> str:
    r"""Format an explanation.

    Normally all embedded newlines are escaped, however there are
    three exceptions: \n{, \n} and \n~.  The first two are intended
    cover nested explanations, see function and attribute explanations
    for examples (.visit_Call(), visit_Attribute()).  The last one is
    for when one explanation needs to span multiple lines, e.g. when
    displaying diffs.
    """
    ...

def issequence(x: Any) -> bool:
    ...

def istext(x: Any) -> bool:
    ...

def isdict(x: Any) -> bool:
    ...

def isset(x: Any) -> bool:
    ...

def isnamedtuple(obj: Any) -> bool:
    ...

def isdatacls(obj: Any) -> bool:
    ...

def isattrs(obj: Any) -> bool:
    ...

def isiterable(obj: Any) -> bool:
    ...

def assertrepr_compare(config, op: str, left: Any, right: Any) -> Optional[List[str]]:
    """Return specialised explanations for some operators/operands."""
    ...

