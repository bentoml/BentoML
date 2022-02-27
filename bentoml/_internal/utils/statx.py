# Modified from https://github.com/ckarageorgkaneen/pystatx
# This will address correct birthtime on Linux system.
# Most implementation comes from https://man7.org/linux/man-pages/man2/statx.2.html#NOTES
# Until Python have a stable API support this, resolve to this.

import ctypes
import typing as t
import logging
import platform
from typing import TYPE_CHECKING
from functools import partial
from functools import lru_cache

from bentoml.exceptions import BentoMLException

if TYPE_CHECKING:
    from ..types import PathType

__all__ = ["statx"]

logger = logging.getLogger(__name__)


def get_syscall_number() -> t.Optional[int]:
    # https://unix.stackexchange.com/a/499016
    # https://marcin.juszkiewicz.com.pl/download/tables/syscalls.html
    if platform.system() == "Linux":
        machine = platform.machine()
        if machine == "x86_64":
            return 332
        elif machine.startswith("arm") or machine.startswith("aarch"):
            return 291
        elif machine == "i386" or machine == "i686" or machine == "ppc64le":
            return 383
        return None
    else:
        import_module = ".".join(__file__.replace(".py", "").split("/")[-4:])
        raise BentoMLException(
            f"{import_module} should only be used on Linux system, while currently on {platform.system()} (which supports os.stat().st_birthtime)"
        )


@lru_cache(maxsize=1)
def get_syscall_func() -> "t.Callable[..., t.Any]":
    syscall_nr = get_syscall_number()
    if syscall_nr is None:
        raise BentoMLException(
            "Only x86, arm64, x86_64 and ppc64le machines on Linux are supported."
        )
    syscall = ctypes.CDLL(None).syscall
    syscall.restype = ctypes.c_int
    syscall.argtypes = [
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(StatxStruct),
    ]
    return partial(syscall, syscall_nr)


class StatxStructTimestamp(ctypes.Structure):
    # statx file timestamp C struct.
    _fields_ = [
        ("tv_sec", ctypes.c_longlong),  # Seconds since the Epoch (UNIX time)
        ("tv_nsec", ctypes.c_uint),  # Nanoseconds since tv_sec
        ("__statx_timestamp_pad1", ctypes.c_int * 1),
    ]


class StatxStruct(ctypes.Structure):
    # statx data buffer C struct.

    _fields_ = [
        ("stx_mask", ctypes.c_uint),  # Mask of bits indicating filled fields
        ("stx_blksize", ctypes.c_uint),  # Block size for filesystem I/O
        ("stx_attributes", ctypes.c_ulonglong),  # Extra file attribute indicators
        ("stx_nlink", ctypes.c_uint),  # Number of hard links
        ("stx_uid", ctypes.c_uint),  # User ID of owner
        ("stx_gid", ctypes.c_uint),  # Group ID of owner
        ("stx_mode", ctypes.c_ushort),  # File type and mode
        ("stx_ino", ctypes.c_ulonglong),  # Inode number
        ("stx_size", ctypes.c_ulonglong),  # Total size in bytes
        ("stx_blocks", ctypes.c_ulonglong),  # Number of 512B blocks allocated
        (
            "stx_attributes_mask",
            ctypes.c_ulonglong,
        ),  # Mask to show what's supported in stx_attributes
        ("__statx_pad1", ctypes.c_ushort * 1),
        # The following fields are file timestamps
        ("stx_atime", StatxStructTimestamp),  # Last access
        ("stx_btime", StatxStructTimestamp),  # Creation
        ("stx_ctime", StatxStructTimestamp),  # Last status change
        ("stx_mtime", StatxStructTimestamp),  # Last modification
        # If this file represents a device, then the next two fields contain the ID of the device
        ("stx_rdev_major", ctypes.c_uint),  # Major ID
        ("stx_rdev_minor", ctypes.c_uint),  # Minor ID
        # The next two fields contain the ID of the device containing the filesystem where the file resides
        ("stx_dev_major", ctypes.c_uint),  # Major ID
        ("stx_dev_minor", ctypes.c_uint),  # Minor ID
        ("__statx_pad2", ctypes.c_ulonglong * 14),
    ]


def _stx_timestamp(struct_statx_timestamp: "StatxStructTimestamp") -> float:
    # Return statx timestamp.
    return struct_statx_timestamp.tv_sec + struct_statx_timestamp.tv_nsec * 1e-9


class _Statx(object):
    # statx system call wrapper class.

    # from kernel-sources include/linux/fcntl.h
    _AT_SYMLINK_NOFOLLOW = 0x100  # Do not follow symbolic links.
    _AT_NO_AUTOMOUNT = 0x800  # Suppress terminal automount traversal
    _AT_FDCWD = -100  # Special value used to indicate

    # openat should use the current working directory.
    _AT_STATX_SYNC_TYPE = 0x6000  # Type of synchronisation required from statx
    _AT_STATX_FORCE_SYNC = 0x2000  # Force the attributes to be sync'd with the server
    _AT_STATX_DONT_SYNC = 0x4000  # Don't sync attributes with the server

    # from kernel-sources include/uapi/linux/stat.h
    _S_IFMT = 0o170000
    _S_IFSOCK = 0o140000
    _S_IFLNK = 0o120000
    _S_IFREG = 0o100000
    _S_IFBLK = 0o60000
    _S_IFDIR = 0o40000
    _S_IFCHR = 0o20000
    _S_IFIFO = 0o10000
    _STATX_TYPE = 0x00000001  # Want/got stx_mode & _S_IFMT
    _STATX_MODE = 0x00000002  # Want/got stx_mode & ~_S_IFMT
    _STATX_NLINK = 0x00000004  # Want/got stx_nlink
    _STATX_UID = 0x00000008  # Want/got stx_uid
    _STATX_GID = 0x00000010  # Want/got stx_gid
    _STATX_ATIME = 0x00000020  # Want/got stx_atime
    _STATX_MTIME = 0x00000040  # Want/got stx_mtime
    _STATX_CTIME = 0x00000080  # Want/got stx_ctime
    _STATX_INO = 0x00000100  # Want/got stx_ino
    _STATX_SIZE = 0x00000200  # Want/got stx_size
    _STATX_BLOCKS = 0x00000400  # Want/got stx_blocks
    _STATX_BASIC_STATS = 0x000007FF  # The stuff in the normal stat struct
    _STATX_BTIME = 0x00000800  # Want/got stx_btime
    _STATX_ALL = 0x00000FFF  # All currently supported flags

    def __init__(
        self,
        filepath: "PathType",
        no_automount: bool = False,
        follow_symlinks: bool = True,
        get_basic_stats: bool = False,
        get_filesize_only: bool = False,
        force_sync: bool = False,
        dont_sync: bool = False,
    ):
        """Construct statx wrapper class."""
        statx_syscall = get_syscall_func()
        self._dirfd = self._AT_FDCWD
        self._flag = self._AT_SYMLINK_NOFOLLOW
        self._mask = self._STATX_ALL
        if no_automount:
            self._flag |= self._AT_NO_AUTOMOUNT
        if follow_symlinks:
            self._flag &= ~self._AT_SYMLINK_NOFOLLOW
        if get_basic_stats:
            self._mask &= ~self._STATX_BASIC_STATS
        if get_filesize_only:
            self._mask = self._STATX_SIZE
        if force_sync:
            self._flag &= ~self._AT_STATX_SYNC_TYPE
            self._flag |= self._AT_STATX_FORCE_SYNC
        if dont_sync:
            self._flag &= ~self._AT_STATX_SYNC_TYPE
            self._flag |= self._AT_STATX_DONT_SYNC
        self._struct_statx_buf = StatxStruct()
        statx_syscall(
            self._dirfd,
            filepath.encode(),
            self._flag,
            self._mask,
            self._struct_statx_buf,
        )

    @property
    def mask(self):
        """Return mask of bits indicating filled fields."""
        return self._struct_statx_buf.stx_mask

    @property
    def btime(self) -> t.Optional[float]:
        """Return the birth time."""
        if self.mask & self._STATX_BTIME:
            return _stx_timestamp(self._struct_statx_buf.stx_btime)
        return None

    @property
    def ctime(self):
        """Return the change time."""
        if self.mask & self._STATX_CTIME:
            return _stx_timestamp(self._struct_statx_buf.stx_ctime)
        return None

    @property
    def mtime(self):
        """Return the modification time."""
        if self.mask & self._STATX_MTIME:
            return _stx_timestamp(self._struct_statx_buf.stx_mtime)
        return None


def statx(
    filepath: "PathType",
    no_automount: bool = False,
    follow_symlinks: bool = True,
    get_basic_stats: bool = False,
    get_filesize_only: bool = False,
    force_sync: bool = False,
    dont_sync: bool = False,
):
    """Return statx data buffer object."""
    return _Statx(
        filepath,
        no_automount=no_automount,
        follow_symlinks=follow_symlinks,
        get_basic_stats=get_basic_stats,
        get_filesize_only=get_filesize_only,
        force_sync=force_sync,
        dont_sync=dont_sync,
    )
