

from enum import IntEnum, unique

"""Enums used by PyFilesystem.
"""

@unique
class ResourceType(IntEnum):
    """Resource Types.

    Positive values are reserved, negative values are implementation
    dependent.

    Most filesystems will support only directory(1) and file(2). Other
    types exist to identify more exotic resource types supported
    by Linux filesystems.

    """

    unknown = ...
    directory = ...
    file = ...
    character = ...
    block_special_file = ...
    fifo = ...
    socket = ...
    symlink = ...

@unique
class Seek(IntEnum):
    """Constants used by `io.IOBase.seek`.

    These match `os.SEEK_CUR`, `os.SEEK_END`, and `os.SEEK_SET`
    from the standard library.

    """

    current = ...
    end = ...
    set = ...
