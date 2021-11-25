

import typing
from typing import Optional, Text

import six

"""Exception classes thrown by filesystem operations.

Errors relating to the underlying filesystem are translated in
to one of the following exceptions.

All Exception classes are derived from `~fs.errors.FSError`
which may be used as a catch-all filesystem exception.

"""
if typing.TYPE_CHECKING: ...
__all__ = [
    "BulkCopyFailed",
    "CreateFailed",
    "DestinationExists",
    "DirectoryExists",
    "DirectoryExpected",
    "DirectoryNotEmpty",
    "FileExists",
    "FileExpected",
    "FilesystemClosed",
    "FSError",
    "IllegalBackReference",
    "InsufficientStorage",
    "InvalidCharsInPath",
    "InvalidPath",
    "MissingInfoNamespace",
    "NoSysPath",
    "NoURL",
    "OperationFailed",
    "OperationTimeout",
    "PathError",
    "PermissionDenied",
    "RemoteConnectionError",
    "RemoveRootError",
    "ResourceError",
    "ResourceInvalid",
    "ResourceLocked",
    "ResourceNotFound",
    "ResourceReadOnly",
    "Unsupported",
]

class MissingInfoNamespace(AttributeError):
    """An expected namespace is missing."""

    def __init__(self, namespace: Text) -> None: ...
    def __reduce__(self): ...

@six.python_2_unicode_compatible
class FSError(Exception):
    """Base exception for the `fs` module."""

    default_message = ...
    def __init__(self, msg: Optional[Text] = ...) -> None: ...
    def __str__(self) -> Text:
        """Return the error message."""
        ...
    def __repr__(self) -> Text: ...

class FilesystemClosed(FSError):
    """Attempt to use a closed filesystem."""

    default_message = ...

class BulkCopyFailed(FSError):
    """A copy operation failed in worker threads."""

    default_message = ...
    def __init__(self, errors) -> None: ...

class CreateFailed(FSError):
    """Filesystem could not be created."""

    default_message = ...
    def __init__(
        self, msg: Optional[Text] = ..., exc: Optional[Exception] = ...
    ) -> None: ...
    @classmethod
    def catch_all(cls, func): ...
    def __reduce__(self): ...

class PathError(FSError):
    """Base exception for errors to do with a path string."""

    default_message = ...
    def __init__(
        self, path: Text, msg: Optional[Text] = ..., exc: Optional[Exception] = ...
    ) -> None: ...
    def __reduce__(self): ...

class NoSysPath(PathError):
    """The filesystem does not provide *sys paths* to the resource."""

    default_message = ...

class NoURL(PathError):
    """The filesystem does not provide an URL for the resource."""

    default_message = ...
    def __init__(
        self, path: Text, purpose: Text, msg: Optional[Text] = ...
    ) -> None: ...
    def __reduce__(self): ...

class InvalidPath(PathError):
    """Path can't be mapped on to the underlaying filesystem."""

    default_message = ...

class InvalidCharsInPath(InvalidPath):
    """Path contains characters that are invalid on this filesystem."""

    default_message = ...

class OperationFailed(FSError):
    """A specific operation failed."""

    default_message = ...
    def __init__(
        self,
        path: Optional[Text] = ...,
        exc: Optional[Exception] = ...,
        msg: Optional[Text] = ...,
    ) -> None: ...
    def __reduce__(self): ...

class Unsupported(OperationFailed):
    """Operation not supported by the filesystem."""

    default_message = ...

class RemoteConnectionError(OperationFailed):
    """Operations encountered remote connection trouble."""

    default_message = ...

class InsufficientStorage(OperationFailed):
    """Storage is insufficient for requested operation."""

    default_message = ...

class PermissionDenied(OperationFailed):
    """Not enough permissions."""

    default_message = ...

class OperationTimeout(OperationFailed):
    """Filesystem took too long."""

    default_message = ...

class RemoveRootError(OperationFailed):
    """Attempt to remove the root directory."""

    default_message = ...

class ResourceError(FSError):
    """Base exception class for error associated with a specific resource."""

    default_message = ...
    def __init__(
        self, path: Text, exc: Optional[Exception] = ..., msg: Optional[Text] = ...
    ) -> None: ...
    def __reduce__(self): ...

class ResourceNotFound(ResourceError):
    """Required resource not found."""

    default_message = ...

class ResourceInvalid(ResourceError):
    """Resource has the wrong type."""

    default_message = ...

class FileExists(ResourceError):
    """File already exists."""

    default_message = ...

class FileExpected(ResourceInvalid):
    """Operation only works on files."""

    default_message = ...

class DirectoryExpected(ResourceInvalid):
    """Operation only works on directories."""

    default_message = ...

class DestinationExists(ResourceError):
    """Target destination already exists."""

    default_message = ...

class DirectoryExists(ResourceError):
    """Directory already exists."""

    default_message = ...

class DirectoryNotEmpty(ResourceError):
    """Attempt to remove a non-empty directory."""

    default_message = ...

class ResourceLocked(ResourceError):
    """Attempt to use a locked resource."""

    default_message = ...

class ResourceReadOnly(ResourceError):
    """Attempting to modify a read-only resource."""

    default_message = ...

class IllegalBackReference(ValueError):
    """Too many backrefs exist in a path.

    This error will occur if the back references in a path would be
    outside of the root. For example, ``"/foo/../../"``, contains two back
    references which would reference a directory above the root.

    Note:
        This exception is a subclass of `ValueError` as it is not
        strictly speaking an issue with a filesystem or resource.

    """

    def __init__(self, path: Text) -> None: ...
    def __reduce__(self): ...

class UnsupportedHash(ValueError):
    """The requested hash algorithm is not supported.

    This exception will be thrown if a hash algorithm is requested that is
    not supported by hashlib.

    """

    ...
