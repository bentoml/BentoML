"""
Disk management utilities.
"""

def disk_used(path):  # -> int:
    """Return the disk usage in a directory."""
    ...

def memstr_to_bytes(text):  # -> int:
    """Convert a memory text to its value in bytes."""
    ...

def mkdirp(d):  # -> None:
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    ...

RM_SUBDIRS_RETRY_TIME = ...
RM_SUBDIRS_N_RETRY = ...

def rm_subdirs(path, onerror=...):  # -> None:
    """Remove all subdirectories in this path.

    The directory indicated by `path` is left in place, and its subdirectories
    are erased.

    If onerror is set, it is called to handle the error with arguments (func,
    path, exc_info) where func is os.listdir, os.remove, or os.rmdir;
    path is the argument to that function that caused it to fail; and
    exc_info is a tuple returned by sys.exc_info().  If onerror is None,
    an exception is raised.
    """
    ...

def delete_folder(folder_path, onerror=..., allow_non_empty=...):
    """Utility function to cleanup a temporary folder if it still exists."""
    ...
