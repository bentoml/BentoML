"""
This type stub file was generated by pyright.
"""

import ctypes
import sys

SEM_FAILURE = ...
if sys.platform == 'darwin':
    ...
RECURSIVE_MUTEX = ...
SEMAPHORE = ...
SEM_OFLAG = ...
SEM_PERM = ...
class timespec(ctypes.Structure):
    _fields_ = ...


if sys.platform != 'win32':
    pthread = ...
if sys.version_info[: 2] < (3, 3):
    class FileExistsError(OSError):
        ...
    
    
    class FileNotFoundError(OSError):
        ...
    
    
def sem_unlink(name): # -> None:
    ...

class SemLock:
    """ctypes wrapper to the unix semaphore"""
    _rand = ...
    def __init__(self, kind, value, maxvalue, name=..., unlink_now=...) -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def acquire(self, block=..., timeout=...):
        ...
    
    def release(self): # -> None:
        ...
    


def raiseFromErrno(): # -> NoReturn:
    ...

