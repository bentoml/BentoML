
import os
import sys
from enum import Enum
from pathlib import Path, PurePath
from types import ModuleType
from typing import Callable, Iterable, Iterator, Optional, Set, Union

import py

LOCK_TIMEOUT = ...
_AnyPurePath = ...
_IGNORED_ERRORS = ...
_IGNORED_WINERRORS = ...
def get_lock_path(path: _AnyPurePath) -> _AnyPurePath:
    ...

def on_rm_rf_error(func, path: str, exc, *, start_path: Path) -> bool:
    """Handle known read-only errors during rmtree.

    The returned value is used only by our own tests.
    """
    ...

def ensure_extended_length_path(path: Path) -> Path:
    """Get the extended-length version of a path (Windows).

    On Windows, by default, the maximum length of a path (MAX_PATH) is 260
    characters, and operations on paths longer than that fail. But it is possible
    to overcome this by converting the path to "extended-length" form before
    performing the operation:
    https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file#maximum-path-length-limitation

    On Windows, this function returns the extended-length absolute version of path.
    On other platforms it returns path unchanged.
    """
    ...

def get_extended_length_path_str(path: str) -> str:
    """Convert a path to a Windows extended length path."""
    ...

def rm_rf(path: Path) -> None:
    """Remove the path contents recursively, even if some elements
    are read-only."""
    ...

def find_prefixed(root: Path, prefix: str) -> Iterator[Path]:
    """Find all elements in root that begin with the prefix, case insensitive."""
    ...

def extract_suffixes(iter: Iterable[PurePath], prefix: str) -> Iterator[str]:
    """Return the parts of the paths following the prefix.

    :param iter: Iterator over path names.
    :param prefix: Expected prefix of the path names.
    """
    ...

def find_suffixes(root: Path, prefix: str) -> Iterator[str]:
    """Combine find_prefixes and extract_suffixes."""
    ...

def parse_num(maybe_num) -> int:
    """Parse number path suffixes, returns -1 on error."""
    ...

def make_numbered_dir(root: Path, prefix: str, mode: int = ...) -> Path:
    """Create a directory with an increased number as suffix for the given prefix."""
    ...

def create_cleanup_lock(p: Path) -> Path:
    """Create a lock to prevent premature folder cleanup."""
    ...

def register_cleanup_lock_removal(lock_path: Path, register=...):
    """Register a cleanup function for removing a lock, by default on atexit."""
    ...

def maybe_delete_a_numbered_dir(path: Path) -> None:
    """Remove a numbered directory if its lock can be obtained and it does
    not seem to be in use."""
    ...

def ensure_deletable(path: Path, consider_lock_dead_if_created_before: float) -> bool:
    """Check if `path` is deletable based on whether the lock file is expired."""
    ...

def try_cleanup(path: Path, consider_lock_dead_if_created_before: float) -> None:
    """Try to cleanup a folder if we can ensure it's deletable."""
    ...

def cleanup_candidates(root: Path, prefix: str, keep: int) -> Iterator[Path]:
    """List candidates for numbered directories to be removed - follows py.path."""
    ...

def cleanup_numbered_dir(root: Path, prefix: str, keep: int, consider_lock_dead_if_created_before: float) -> None:
    """Cleanup for lock driven numbered directories."""
    ...

def make_numbered_dir_with_cleanup(root: Path, prefix: str, keep: int, lock_timeout: float, mode: int) -> Path:
    """Create a numbered dir with a cleanup lock and remove old ones."""
    ...

def resolve_from_str(input: str, rootpath: Path) -> Path:
    ...

def fnmatch_ex(pattern: str, path) -> bool:
    """A port of FNMatcher from py.path.common which works with PurePath() instances.

    The difference between this algorithm and PurePath.match() is that the
    latter matches "**" glob expressions for each part of the path, while
    this algorithm uses the whole path instead.

    For example:
        "tests/foo/bar/doc/test_foo.py" matches pattern "tests/**/doc/test*.py"
        with this algorithm, but not with PurePath.match().

    This algorithm was ported to keep backward-compatibility with existing
    settings which assume paths match according this logic.

    References:
    * https://bugs.python.org/issue29249
    * https://bugs.python.org/issue34731
    """
    ...

def parts(s: str) -> Set[str]:
    ...

def symlink_or_skip(src, dst, **kwargs): # -> None:
    """Make a symlink, or skip the test in case symlinks are not supported."""
    ...

class ImportMode(Enum):
    """Possible values for `mode` parameter of `import_path`."""
    prepend = ...
    append = ...
    importlib = ...


class ImportPathMismatchError(ImportError):
    """Raised on import_path() if there is a mismatch of __file__'s.

    This can happen when `import_path` is called multiple times with different filenames that has
    the same basename but reside in packages
    (for example "/tests1/test_foo.py" and "/tests2/test_foo.py").
    """
    ...


def import_path(p: Union[str, py.path.local, Path], *, mode: Union[str, ImportMode] = ...) -> ModuleType:
    """Import and return a module from the given path, which can be a file (a module) or
    a directory (a package).

    The import mechanism used is controlled by the `mode` parameter:

    * `mode == ImportMode.prepend`: the directory containing the module (or package, taking
      `__init__.py` files into account) will be put at the *start* of `sys.path` before
      being imported with `__import__.

    * `mode == ImportMode.append`: same as `prepend`, but the directory will be appended
      to the end of `sys.path`, if not already in `sys.path`.

    * `mode == ImportMode.importlib`: uses more fine control mechanisms provided by `importlib`
      to import the module, which avoids having to use `__import__` and muck with `sys.path`
      at all. It effectively allows having same-named test modules in different places.

    :raises ImportPathMismatchError:
        If after importing the given `path` and the module `__file__`
        are different. Only raised in `prepend` and `append` modes.
    """
    ...

if sys.platform.startswith("win"):
    ...
else:
    ...
def resolve_package_path(path: Path) -> Optional[Path]:
    """Return the Python package path by looking for the last
    directory upwards which still contains an __init__.py.

    Returns None if it can not be determined.
    """
    ...

def visit(path: str, recurse: Callable[[os.DirEntry[str]], bool]) -> Iterator[os.DirEntry[str]]:
    """Walk a directory recursively, in breadth-first order.

    Entries at each directory level are sorted.
    """
    ...

def absolutepath(path: Union[Path, str]) -> Path:
    """Convert a path to an absolute path using os.path.abspath.

    Prefer this over Path.resolve() (see #6523).
    Prefer this over Path.absolute() (not public, doesn't normalize).
    """
    ...

def commonpath(path1: Path, path2: Path) -> Optional[Path]:
    """Return the common part shared with the other path, or None if there is
    no common part.

    If one path is relative and one is absolute, returns None.
    """
    ...

def bestrelpath(directory: Path, dest: Path) -> str:
    """Return a string which is a relative path from directory to dest such
    that directory/bestrelpath == dest.

    The paths must be either both absolute or both relative.

    If no such path can be determined, returns dest.
    """
    ...

