
from typing import Optional, TextIO

from _pytest.compat import final

"""Helper functions for writing to terminals and files."""
def get_terminal_width() -> int:
    ...

def should_do_markup(file: TextIO) -> bool:
    ...

@final
class TerminalWriter:
    _esctable = ...
    def __init__(self, file: Optional[TextIO] = ...) -> None:
        ...
    
    @property
    def fullwidth(self) -> int:
        ...
    
    @fullwidth.setter
    def fullwidth(self, value: int) -> None:
        ...
    
    @property
    def width_of_current_line(self) -> int:
        """Return an estimate of the width so far in the current line."""
        ...
    
    def markup(self, text: str, **markup: bool) -> str:
        ...
    
    def sep(self, sepchar: str, title: Optional[str] = ..., fullwidth: Optional[int] = ..., **markup: bool) -> None:
        ...
    
    def write(self, msg: str, *, flush: bool = ..., **markup: bool) -> None:
        ...
    
    def line(self, s: str = ..., **markup: bool) -> None:
        ...
    
    def flush(self) -> None:
        ...
    


