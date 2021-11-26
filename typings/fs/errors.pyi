import typing
from typing import Optional, Text
import six

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
    def __init__(self, namespace: Text) -> None: ...
    def __reduce__(self): ...

@six.python_2_unicode_compatible
class FSError(Exception):
    default_message = ...
    def __init__(self, msg: Optional[Text] = ...) -> None: ...
    def __str__(self) -> Text: ...
    def __repr__(self) -> Text: ...

class FilesystemClosed(FSError):
    default_message = ...

class BulkCopyFailed(FSError):
    default_message = ...
    def __init__(self, errors) -> None: ...

class CreateFailed(FSError):
    default_message = ...
    def __init__(
        self, msg: Optional[Text] = ..., exc: Optional[Exception] = ...
    ) -> None: ...
    @classmethod
    def catch_all(cls, func): ...
    def __reduce__(self): ...

class PathError(FSError):
    default_message = ...
    def __init__(
        self, path: Text, msg: Optional[Text] = ..., exc: Optional[Exception] = ...
    ) -> None: ...
    def __reduce__(self): ...

class NoSysPath(PathError):
    default_message = ...

class NoURL(PathError):
    default_message = ...
    def __init__(
        self, path: Text, purpose: Text, msg: Optional[Text] = ...
    ) -> None: ...
    def __reduce__(self): ...

class InvalidPath(PathError):
    default_message = ...

class InvalidCharsInPath(InvalidPath):
    default_message = ...

class OperationFailed(FSError):
    default_message = ...
    def __init__(
        self,
        path: Optional[Text] = ...,
        exc: Optional[Exception] = ...,
        msg: Optional[Text] = ...,
    ) -> None: ...
    def __reduce__(self): ...

class Unsupported(OperationFailed):
    default_message = ...

class RemoteConnectionError(OperationFailed):
    default_message = ...

class InsufficientStorage(OperationFailed):
    default_message = ...

class PermissionDenied(OperationFailed):
    default_message = ...

class OperationTimeout(OperationFailed):
    default_message = ...

class RemoveRootError(OperationFailed):
    default_message = ...

class ResourceError(FSError):
    default_message = ...
    def __init__(
        self, path: Text, exc: Optional[Exception] = ..., msg: Optional[Text] = ...
    ) -> None: ...
    def __reduce__(self): ...

class ResourceNotFound(ResourceError):
    default_message = ...

class ResourceInvalid(ResourceError):
    default_message = ...

class FileExists(ResourceError):
    default_message = ...

class FileExpected(ResourceInvalid):
    default_message = ...

class DirectoryExpected(ResourceInvalid):
    default_message = ...

class DestinationExists(ResourceError):
    default_message = ...

class DirectoryExists(ResourceError):
    default_message = ...

class DirectoryNotEmpty(ResourceError):
    default_message = ...

class ResourceLocked(ResourceError):
    default_message = ...

class ResourceReadOnly(ResourceError):
    default_message = ...

class IllegalBackReference(ValueError):
    def __init__(self, path: Text) -> None: ...
    def __reduce__(self): ...

class UnsupportedHash(ValueError): ...
