

import typing
from contextlib import contextmanager
from types import TracebackType
from typing import Iterator, Optional, Text, Type, Union

"""Tools for managing OS errors.
"""
if typing.TYPE_CHECKING: ...
_WINDOWS_PLATFORM = ...

class _ConvertOSErrors:
    """Context manager to convert OSErrors in to FS Errors."""

    FILE_ERRORS = ...
    DIR_ERRORS = ...
    if _WINDOWS_PLATFORM: ...
    def __init__(self, opname: Text, path: Text, directory: bool = ...) -> None: ...
    def __enter__(self) -> _ConvertOSErrors: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...

convert_os_errors = _ConvertOSErrors

@contextmanager
def unwrap_errors(path_replace: Union[Text, Mapping[Text, Text]]) -> Iterator[None]:
    """Get a context to map OS errors to their `fs.errors` counterpart.

    The context will re-write the paths in resource exceptions to be
    in the same context as the wrapped filesystem.

    The only parameter may be the path from the parent, if only one path
    is to be unwrapped. Or it may be a dictionary that maps wrapped
    paths on to unwrapped paths.

    """
    ...
