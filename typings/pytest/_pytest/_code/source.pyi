
import ast
import types
from typing import Iterable, Iterator, List, Optional, Tuple, Union, overload

class Source:
    """An immutable object holding a source code fragment.

    When using Source(...), the source lines are deindented.
    """
    def __init__(self, obj: object = ...) -> None:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    __hash__ = ...
    @overload
    def __getitem__(self, key: int) -> str:
        ...
    
    @overload
    def __getitem__(self, key: slice) -> Source:
        ...
    
    def __getitem__(self, key: Union[int, slice]) -> Union[str, Source]:
        ...
    
    def __iter__(self) -> Iterator[str]:
        ...
    
    def __len__(self) -> int:
        ...
    
    def strip(self) -> Source:
        """Return new Source object with trailing and leading blank lines removed."""
        ...
    
    def indent(self, indent: str = ...) -> Source:
        """Return a copy of the source object with all lines indented by the
        given indent-string."""
        ...
    
    def getstatement(self, lineno: int) -> Source:
        """Return Source statement which contains the given linenumber
        (counted from 0)."""
        ...
    
    def getstatementrange(self, lineno: int) -> Tuple[int, int]:
        """Return (start, end) tuple which spans the minimal statement region
        which containing the given lineno."""
        ...
    
    def deindent(self) -> Source:
        """Return a new Source object deindented."""
        ...
    
    def __str__(self) -> str:
        ...
    


def findsource(obj) -> Tuple[Optional[Source], int]:
    ...

def getrawcode(obj: object, trycall: bool = ...) -> types.CodeType:
    """Return code object for given function."""
    ...

def deindent(lines: Iterable[str]) -> List[str]:
    ...

def get_statement_startend2(lineno: int, node: ast.AST) -> Tuple[int, Optional[int]]:
    ...

def getstatementrange_ast(lineno: int, source: Source, assertion: bool = ..., astnode: Optional[ast.AST] = ...) -> Tuple[ast.AST, int, int]:
    ...

