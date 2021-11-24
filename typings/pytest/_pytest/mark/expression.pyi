
import ast
import enum
import types
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
)

import attr

r"""Evaluate match expressions, as used by `-k` and `-m`.

The grammar is:

expression: expr? EOF
expr:       and_expr ('or' and_expr)*
and_expr:   not_expr ('and' not_expr)*
not_expr:   'not' not_expr | '(' expr ')' | ident
ident:      (\w|:|\+|-|\.|\[|\])+

The semantics are:

- Empty expression evaluates to False.
- ident evaluates to True of False according to a provided matcher function.
- or/and/not evaluate according to the usual boolean semantics.
"""
if TYPE_CHECKING:
    ...
__all__ = ["Expression", "ParseError"]
class TokenType(enum.Enum):
    LPAREN = ...
    RPAREN = ...
    OR = ...
    AND = ...
    NOT = ...
    IDENT = ...
    EOF = ...


@attr.s(frozen=True, slots=True)
class Token:
    type = ...
    value = ...
    pos = ...


class ParseError(Exception):
    """The expression contains invalid syntax.

    :param column: The column in the line where the error occurred (1-based).
    :param message: A description of the error.
    """
    def __init__(self, column: int, message: str) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    


class Scanner:
    __slots__ = ...
    def __init__(self, input: str) -> None:
        ...
    
    def lex(self, input: str) -> Iterator[Token]:
        ...
    
    def accept(self, type: TokenType, *, reject: bool = ...) -> Optional[Token]:
        ...
    
    def reject(self, expected: Sequence[TokenType]) -> NoReturn:
        ...
    


IDENT_PREFIX = ...
def expression(s: Scanner) -> ast.Expression:
    ...

def expr(s: Scanner) -> ast.expr:
    ...

def and_expr(s: Scanner) -> ast.expr:
    ...

def not_expr(s: Scanner) -> ast.expr:
    ...

class MatcherAdapter(Mapping[str, bool]):
    """Adapts a matcher function to a locals mapping as required by eval()."""
    def __init__(self, matcher: Callable[[str], bool]) -> None:
        ...
    
    def __getitem__(self, key: str) -> bool:
        ...
    
    def __iter__(self) -> Iterator[str]:
        ...
    
    def __len__(self) -> int:
        ...
    


class Expression:
    """A compiled match expression as used by -k and -m.

    The expression can be evaulated against different matchers.
    """
    __slots__ = ...
    def __init__(self, code: types.CodeType) -> None:
        ...
    
    @classmethod
    def compile(self, input: str) -> Expression:
        """Compile a match expression.

        :param input: The input expression - one line.
        """
        ...
    
    def evaluate(self, matcher: Callable[[str], bool]) -> bool:
        """Evaluate the match expression.

        :param matcher:
            Given an identifier, should return whether it matches or not.
            Should be prepared to handle arbitrary strings as input.

        :returns: Whether the expression matches or not.
        """
        ...
    


