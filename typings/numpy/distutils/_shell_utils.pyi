import os

"""
Helper functions for interacting with the shell, and consuming shell-style
parameters provided in config files.
"""
__all__ = ["WindowsParser", "PosixParser", "NativeParser"]

class CommandLineParser:
    """
    An object that knows how to split and join command-line arguments.

    It must be true that ``argv == split(join(argv))`` for all ``argv``.
    The reverse neednt be true - `join(split(cmd))` may result in the addition
    or removal of unnecessary escaping.
    """

    @staticmethod
    def join(argv):
        """Join a list of arguments into a command line string"""
        ...
    @staticmethod
    def split(cmd):
        """Split a command line string into a list of arguments"""
        ...

class WindowsParser:
    """
    The parsing behavior used by `subprocess.call("string")` on Windows, which
    matches the Microsoft C/C++ runtime.

    Note that this is _not_ the behavior of cmd.
    """

    @staticmethod
    def join(argv): ...
    @staticmethod
    def split(cmd): ...

class PosixParser:
    """
    The parsing behavior used by `subprocess.call("string", shell=True)` on Posix.
    """

    @staticmethod
    def join(argv): ...
    @staticmethod
    def split(cmd): ...

if os.name == "nt": ...
else:
    NativeParser = PosixParser
