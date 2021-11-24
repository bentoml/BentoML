
import ast
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

from _pytest.config import Config
from _pytest.main import Session

"""Rewrite assertion AST to produce nice error messages."""
if TYPE_CHECKING:
    ...
assertstate_key = ...
PYTEST_TAG = ...
PYC_EXT = ...
PYC_TAIL = ...
class AssertionRewritingHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """PEP302/PEP451 import hook which rewrites asserts."""
    def __init__(self, config: Config) -> None:
        ...
    
    def set_session(self, session: Optional[Session]) -> None:
        ...
    
    _find_spec = ...
    def find_spec(self, name: str, path: Optional[Sequence[Union[str, bytes]]] = ..., target: Optional[types.ModuleType] = ...) -> Optional[importlib.machinery.ModuleSpec]:
        ...
    
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Optional[types.ModuleType]:
        ...
    
    def exec_module(self, module: types.ModuleType) -> None:
        ...
    
    def mark_rewrite(self, *names: str) -> None:
        """Mark import names as needing to be rewritten.

        The named module or package as well as any nested modules will
        be rewritten on import.
        """
        ...
    
    def get_data(self, pathname: Union[str, bytes]) -> bytes:
        """Optional PEP302 get_data API."""
        ...
    


if sys.platform == "win32":
    ...
else:
    ...
def rewrite_asserts(mod: ast.Module, source: bytes, module_path: Optional[str] = ..., config: Optional[Config] = ...) -> None:
    """Rewrite the assert statements in mod."""
    ...

UNARY_MAP = ...
BINOP_MAP = ...
def set_location(node, lineno, col_offset):
    """Set node location information recursively."""
    ...

class AssertionRewriter(ast.NodeVisitor):
    """Assertion rewriting implementation.

    The main entrypoint is to call .run() with an ast.Module instance,
    this will then find all the assert statements and rewrite them to
    provide intermediate values and a detailed assertion error.  See
    http://pybites.blogspot.be/2011/07/behind-scenes-of-pytests-new-assertion.html
    for an overview of how this works.

    The entry point here is .run() which will iterate over all the
    statements in an ast.Module and for each ast.Assert statement it
    finds call .visit() with it.  Then .visit_Assert() takes over and
    is responsible for creating new ast statements to replace the
    original assert statement: it rewrites the test of an assertion
    to provide intermediate values and replace it with an if statement
    which raises an assertion error with a detailed explanation in
    case the expression is false and calls pytest_assertion_pass hook
    if expression is true.

    For this .visit_Assert() uses the visitor pattern to visit all the
    AST nodes of the ast.Assert.test field, each visit call returning
    an AST node and the corresponding explanation string.  During this
    state is kept in several instance attributes:

    :statements: All the AST statements which will replace the assert
       statement.

    :variables: This is populated by .variable() with each variable
       used by the statements so that they can all be set to None at
       the end of the statements.

    :variable_counter: Counter to create new unique variables needed
       by statements.  Variables are created using .variable() and
       have the form of "@py_assert0".

    :expl_stmts: The AST statements which will be executed to get
       data from the assertion.  This is the code which will construct
       the detailed assertion message that is used in the AssertionError
       or for the pytest_assertion_pass hook.

    :explanation_specifiers: A dict filled by .explanation_param()
       with %-formatting placeholders and their corresponding
       expressions to use in the building of an assertion message.
       This is used by .pop_format_context() to build a message.

    :stack: A stack of the explanation_specifiers dicts maintained by
       .push_format_context() and .pop_format_context() which allows
       to build another %-formatted string while already building one.

    This state is reset on every new assert statement visited and used
    by the other visitors.
    """
    def __init__(self, module_path: Optional[str], config: Optional[Config], source: bytes) -> None:
        ...
    
    def run(self, mod: ast.Module) -> None:
        """Find all assert statements in *mod* and rewrite them."""
        ...
    
    @staticmethod
    def is_rewrite_disabled(docstring: str) -> bool:
        ...
    
    def variable(self) -> str:
        """Get a new variable."""
        ...
    
    def assign(self, expr: ast.expr) -> ast.Name:
        """Give *expr* a name."""
        ...
    
    def display(self, expr: ast.expr) -> ast.expr:
        """Call saferepr on the expression."""
        ...
    
    def helper(self, name: str, *args: ast.expr) -> ast.expr:
        """Call a helper in this module."""
        ...
    
    def builtin(self, name: str) -> ast.Attribute:
        """Return the builtin called *name*."""
        ...
    
    def explanation_param(self, expr: ast.expr) -> str:
        """Return a new named %-formatting placeholder for expr.

        This creates a %-formatting placeholder for expr in the
        current formatting context, e.g. ``%(py0)s``.  The placeholder
        and expr are placed in the current format context so that it
        can be used on the next call to .pop_format_context().
        """
        ...
    
    def push_format_context(self) -> None:
        """Create a new formatting context.

        The format context is used for when an explanation wants to
        have a variable value formatted in the assertion message.  In
        this case the value required can be added using
        .explanation_param().  Finally .pop_format_context() is used
        to format a string of %-formatted values as added by
        .explanation_param().
        """
        ...
    
    def pop_format_context(self, expl_expr: ast.expr) -> ast.Name:
        """Format the %-formatted string with current format context.

        The expl_expr should be an str ast.expr instance constructed from
        the %-placeholders created by .explanation_param().  This will
        add the required code to format said string to .expl_stmts and
        return the ast.Name instance of the formatted string.
        """
        ...
    
    def generic_visit(self, node: ast.AST) -> Tuple[ast.Name, str]:
        """Handle expressions we don't have custom code for."""
        ...
    
    def visit_Assert(self, assert_: ast.Assert) -> List[ast.stmt]:
        """Return the AST statements to replace the ast.Assert instance.

        This rewrites the test of an assertion to provide
        intermediate values and replace it with an if statement which
        raises an assertion error with a detailed explanation in case
        the expression is false.
        """
        ...
    
    def visit_Name(self, name: ast.Name) -> Tuple[ast.Name, str]:
        ...
    
    def visit_BoolOp(self, boolop: ast.BoolOp) -> Tuple[ast.Name, str]:
        ...
    
    def visit_UnaryOp(self, unary: ast.UnaryOp) -> Tuple[ast.Name, str]:
        ...
    
    def visit_BinOp(self, binop: ast.BinOp) -> Tuple[ast.Name, str]:
        ...
    
    def visit_Call(self, call: ast.Call) -> Tuple[ast.Name, str]:
        ...
    
    def visit_Starred(self, starred: ast.Starred) -> Tuple[ast.Starred, str]:
        ...
    
    def visit_Attribute(self, attr: ast.Attribute) -> Tuple[ast.Name, str]:
        ...
    
    def visit_Compare(self, comp: ast.Compare) -> Tuple[ast.expr, str]:
        ...
    


def try_makedirs(cache_dir: Path) -> bool:
    """Attempt to create the given directory and sub-directories exist.

    Returns True if successful or if it already exists.
    """
    ...

def get_cache_dir(file_path: Path) -> Path:
    """Return the cache directory to write .pyc files for the given .py file path."""
    ...

