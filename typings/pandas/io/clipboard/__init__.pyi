import contextlib
import ctypes
import os
import platform
import subprocess
import time
import warnings
from ctypes import c_size_t, c_wchar, c_wchar_p, get_errno, sizeof
from shutil import which

"""
Pyperclip

A cross-platform clipboard module for Python,
with copy & paste functions for plain text.
By Al Sweigart al@inventwithpython.com
BSD License

Usage:
  import pyperclip
  pyperclip.copy('The text to be copied to the clipboard.')
  spam = pyperclip.paste()

  if not pyperclip.is_available():
    print("Copy functionality unavailable!")

On Windows, no additional modules are needed.
On Mac, the pyobjc module is used, falling back to the pbcopy and pbpaste cli
    commands. (These commands should come with OS X.).
On Linux, install xclip or xsel via package manager. For example, in Debian:
    sudo apt-get install xclip
    sudo apt-get install xsel

Otherwise on Linux, you will need the PyQt5 modules installed.

This module does not work with PyGObject yet.

Cygwin is currently not supported.

Security Note: This module runs programs with these names:
    - which
    - where
    - pbcopy
    - pbpaste
    - xclip
    - xsel
    - klipper
    - qdbus
A malicious user could rename or add programs with these names, tricking
Pyperclip into running them with whatever permissions the Python process has.

"""
__version__ = ...
HAS_DISPLAY = ...
EXCEPT_MSG = ...
ENCODING = ...
if platform.system() == "Windows":
    WHICH_CMD = ...
else:
    WHICH_CMD = ...

class PyperclipException(RuntimeError): ...

class PyperclipWindowsException(PyperclipException):
    def __init__(self, message) -> None: ...

def init_osx_pbcopy_clipboard(): ...
def init_osx_pyobjc_clipboard(): ...
def init_qt_clipboard(): ...
def init_xclip_clipboard(): ...
def init_xsel_clipboard(): ...
def init_klipper_clipboard(): ...
def init_dev_clipboard_clipboard(): ...
def init_no_clipboard():  # -> tuple[ClipboardUnavailable, ClipboardUnavailable]:
    class ClipboardUnavailable: ...

class CheckedCall:
    def __init__(self, f) -> None: ...
    def __call__(self, *args): ...
    def __setattr__(self, key, value): ...

def init_windows_clipboard(): ...
def init_wsl_clipboard(): ...
def determine_clipboard():  # -> tuple[(text: Unknown) -> None, () -> str] | tuple[(text: Unknown) -> None, () -> str | None] | tuple[(text: Unknown) -> None, () -> Unknown] | tuple[(text: Unknown, primary: Unknown = False) -> None, (primary: Unknown = False) -> str] | tuple[ClipboardUnavailable, ClipboardUnavailable]:
    """
    Determine the OS/platform and set the copy() and paste() functions
    accordingly.
    """
    ...

def set_clipboard(clipboard):  # -> None:
    """
    Explicitly sets the clipboard mechanism. The "clipboard mechanism" is how
    the copy() and paste() functions interact with the operating system to
    implement the copy/paste feature. The clipboard parameter must be one of:
        - pbcopy
        - pbobjc (default on Mac OS X)
        - qt
        - xclip
        - xsel
        - klipper
        - windows (default on Windows)
        - no (this is what is set when no clipboard mechanism can be found)
    """
    ...

def lazy_load_stub_copy(text):  # -> None:
    """
    A stub function for copy(), which will load the real copy() function when
    called so that the real copy() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
    ...

def lazy_load_stub_paste():  # -> str | None:
    """
    A stub function for paste(), which will load the real paste() function when
    called so that the real paste() function is used for later calls.

    This allows users to import pyperclip without having determine_clipboard()
    automatically run, which will automatically select a clipboard mechanism.
    This could be a problem if it selects, say, the memory-heavy PyQt4 module
    but the user was just going to immediately call set_clipboard() to use a
    different clipboard mechanism.

    The lazy loading this stub function implements gives the user a chance to
    call set_clipboard() to pick another clipboard mechanism. Or, if the user
    simply calls copy() or paste() without calling set_clipboard() first,
    will fall back on whatever clipboard mechanism that determine_clipboard()
    automatically chooses.
    """
    ...

def is_available() -> bool: ...

__all__ = ["copy", "paste", "set_clipboard", "determine_clipboard"]
clipboard_get = ...
clipboard_set = ...
